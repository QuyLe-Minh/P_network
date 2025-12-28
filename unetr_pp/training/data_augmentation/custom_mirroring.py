import numpy as np

def augment_mirroring(sample_data, sample_seg=None, sample_pos=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
        if sample_pos is not None:
            x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = sample_pos
            sample_pos = [x_ub, x_lb, y_lb, y_ub, z_lb, z_ub]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
        if sample_pos is not None:
            x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = sample_pos
            sample_pos = [x_lb, x_ub, y_ub, y_lb, z_lb, z_ub]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
            if sample_pos is not None:
                x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = sample_pos
                sample_pos = [x_lb, x_ub, y_lb, y_ub, z_ub, z_lb]
    return sample_data, sample_seg, np.array(sample_pos)