import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image
from unetr_pp.utilities.random_stuff import no_op
from unetr_pp.utilities.to_torch import to_cuda, maybe_to_torch
from torch import nn
import torch
from scipy.ndimage.filters import gaussian_filter
from typing import Union, Tuple, List


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        return torch.device('cuda')

    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError


class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # if we have 5 pooling then our patch size must be divisible by 2**5
        self.input_shape_must_be_divisible_by = None  # for example in a 2d network that does 5 pool in x and 6 pool
        # in y this would be (32, 64)

        # we need to know this because we need to know if we are a 2d or a 3d netowrk
        self.conv_op = None  # nn.Conv2d or nn.Conv3d

        # this tells us how many channely we have in the output. Important for preallocation in inference
        self.num_classes = None  # number of channels in the output

        # depending on the loss, we do not hard code a nonlinearity into the architecture. To aggregate predictions
        # during inference, we need to apply the nonlinearity, however. So it is important to let the newtork know what
        # to apply in inference. For the most part this will be softmax
        self.inference_apply_nonlin = lambda x: x  # softmax_helper

        # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
        # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
        # can be expensive, so it makes sense to save and reuse them.
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None
        
        # For encoding the position of the patches
        self.posenc = None

    def predict_3D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        assert self.get_device() != "cpu", "CPU not implemented"

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        self.conv_op = nn.Conv3d
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        context = no_op
        
        with context():
            with torch.no_grad():
                res = self._internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                            regions_class_order, use_gaussian, pad_border_mode,
                                                            pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                            verbose=verbose)

        return res

    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map

    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps

    def _internal_predict_3D_3Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
            
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"
        assert self.get_device() != "cpu"
        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        use_gaussian = True
        if use_gaussian:
            print("ALLOW GAUSSIAN")
        else:
            print("DISABLE GAUSSIAN")
        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map).cuda(self.get_device(),
                                                                                     non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones((14, 64, 128, 128), device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones((14, 64, 128, 128), dtype=np.float32)
            
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        c, d, h, w = data_shape
        # threshold = (data.max() - data.min()) * 0.005 + data.min()
        # bitmap = data[0].copy()
        # bitmap[bitmap > threshold] = 1
        # bitmap[bitmap <= threshold] = 0

        # x1, x2, y1, y2 = 0, 0, 0, 0
        # for i in range(bitmap.shape[1]):
        #     if bitmap[int(0.5*d), i, :].sum() >= w*0.1:
        #         x1 = i
        #         break
            
        # for i in range(bitmap.shape[1]-1, -1, -1):
        #     if bitmap[int(0.5*d), i, :].sum() >= w*0.1:
        #         x2 = i
        #         break
        
        # for i in range(bitmap.shape[2]):
        #     if bitmap[int(0.5*d), :, i].sum() >= w*0.1:
        #         y1 = i
        #         break
        
        # for i in range(bitmap.shape[2]-1, -1, -1):
        #     if bitmap[int(0.5*d), :, i].sum() >= w*0.1:
        #         y2 = i
        #         break

        # print(data.shape, x1, x2, y1, y2)
                
        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]
                    
                    # pos = np.array((lb_x/d, ub_x/d, (lb_y-x1)/(x2-x1), (ub_y-x1)/(x2-x1), (lb_z-y1)/(y2-y1), (ub_z-y1)/(y2-y1)))
                    pos = np.array((lb_x/d, ub_x/d, lb_y/w, ub_y/w, lb_z/h, ub_z/h))
                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], pos[None, ...], mirror_axes, do_mirroring,
                        gaussian_importance_map)[0]

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities


    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], pos: np.ndarray, mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'
        # everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        # we now return a cuda tensor! Not numpy array!

        x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]),
                                   dtype=torch.float).cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        if self.posenc is not None:
            pos = self.posenc(pos, x.shape[-3:])
        else:
            # print("WARNING!! You are not using positional encoding!")
            pos = None

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x, pos, is_posenc=True))
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                if pos is None:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, )), pos, is_posenc=True))
                else:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, )), torch.flip(pos, (4, )), is_posenc=True))
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                if pos is None:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3, )), pos, is_posenc=True))
                else:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3, )), torch.flip(pos, (3, )), is_posenc=True))
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                if pos is None:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3)), pos, is_posenc=True))
                else:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3)), torch.flip(pos, (4, 3)), is_posenc=True))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                if pos is None:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (2, )), pos, is_posenc=True))
                else:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (2, )), torch.flip(pos, (2, )), is_posenc=True))
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                if pos is None:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2)), pos, is_posenc=True))
                else:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2)), torch.flip(pos, (4, 2)), is_posenc=True))
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                if pos is None:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2)), pos, is_posenc=True))
                else:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2)), torch.flip(pos, (3, 2)), is_posenc=True))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                if pos is None:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2)), pos, is_posenc=True))
                else:
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2)), torch.flip(pos, (4, 3, 2)), is_posenc=True))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult
        
        return result_torch





