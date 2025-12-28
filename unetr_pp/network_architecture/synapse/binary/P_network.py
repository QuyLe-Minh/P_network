import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Union
from unetr_pp.network_architecture.neural_network import SegmentationNetwork
from unetr_pp.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from unetr_pp.network_architecture.synapse.binary.model_components import UnetrPPEncoder, UnetrUpBlock, PosEnc

class BankTrainer(nn.Module):
    def __init__(self, n_classes=14, feature_size=16, bank_shape=[14, 64, 16], n_samples=64):
        super(BankTrainer, self).__init__()
        
        assert bank_shape[0] == n_classes, "Bank shape should be equal to number of classes"
        
        self.feature_size = feature_size
        self.bank_shape = bank_shape
        self.n_samples = n_samples
        self.n_classes = n_classes

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size*n_samples, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
    def random_sampling(self, gt):
        # Flatten gt and sample efficiently
        sampled_indices = []
        for cls in range(self.n_classes):
            cls_indices = (gt == cls).nonzero(as_tuple=True)[0]
            if len(cls_indices) == 0:
                sampled_indices.append([-1] * self.n_samples)
            else:
                cls_indices = cls_indices[torch.randint(0, len(cls_indices), (self.n_samples,))]
                sampled_indices.append(cls_indices.tolist())
        return sampled_indices
        
    def forward(self, gt, features, bank):
        """
        Forward pass for the network.
        Args:
            gt (torch.Tensor): Ground truth tensor of shape (B, H, W, D) with long type, range from 0 to 13.
            features (torch.Tensor): Feature tensor of shape (B, C, H, W, D).
            bank (torch.Tensor): Bank tensor of shape (N_classes, 64, C).
        Returns:
            torch.Tensor: Computed loss value.
        """
        # Flatten gt and features
        gt = gt.view(-1)
        features = features.permute(0, 2, 3, 4, 1).reshape(-1, self.feature_size)

        # Efficient sampling and selection
        sampled_indices = self.random_sampling(gt)
        selected_features = torch.stack([features[indices] if indices[0] != -1 else bank[cls]
                                        for cls, indices in enumerate(sampled_indices)]).to(features.device)
        
        # Normalize selected features and bank
        flatten_features = selected_features.view(self.n_classes, -1)
        flatten_bank = bank.view(self.n_classes, -1)

        flatten_features = self.linear(flatten_features)
        flatten_bank = self.linear(flatten_bank)

        normalized_features = F.normalize(flatten_features, dim=-1)
        normalized_bank = F.normalize(flatten_bank, dim=-1)
        
        # Compute similarities efficiently
        similarities = (normalized_features @ normalized_bank.T) / 0.5  # n_classes, n_classes

        diagonal = similarities.diag().unsqueeze(1) #N,1
        exp_diagonal = torch.exp(diagonal)
        off_diagonal = torch.exp(similarities).sum(dim=1, keepdim=True) #N,1
        off_diagonal_T = torch.exp(similarities.T).sum(dim=1, keepdim=True) #N,1

        loss = torch.log(exp_diagonal / off_diagonal) + torch.log(exp_diagonal / off_diagonal_T)
        return -1/2*loss.mean()


class P_Network(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [64, 128, 128],
            feature_size: int = 16,
            num_heads: int = 4,
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::

            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        self.img_size = img_size
        
        self.bank = nn.Parameter(torch.empty(out_channels, 64, feature_size), requires_grad=True)
        nn.init.xavier_uniform_(self.bank)

        self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads)
        self.posenc = PosEnc()
        self.bank_trainer = BankTrainer(n_classes=out_channels, feature_size=feature_size, bank_shape=self.bank.shape, n_samples=self.bank.shape[1])

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * 8
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * 16
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * 32,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=64 * 128 * 128,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)

        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def forward(self, x_in, pos, is_posenc=False, gt=None):
        if not is_posenc:
            posenc = self.posenc(pos, self.img_size)
        else:
            posenc = pos
        
        _, hidden_states = self.unetr_pp_encoder(x_in, self.bank)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0] #/2, /4, /4
        enc2 = hidden_states[1] #/4, /8, /8
        enc3 = hidden_states[2] #/8, /16, /16
        enc4 = hidden_states[3]

        # Four decoders
        dec3 = self.decoder5(enc4, enc3, F.interpolate(posenc, scale_factor=(1/8, 1/16, 1/16)), self.bank)
        dec2 = self.decoder4(dec3, enc2, F.interpolate(posenc, scale_factor=(1/4, 1/8, 1/8)), self.bank)
        dec1 = self.decoder3(dec2, enc1, F.interpolate(posenc, scale_factor=(1/2, 1/4, 1/4)), self.bank)
        out = self.decoder2(dec1, convBlock, posenc)
        
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)
            
        if gt is not None:
            loss = self.bank_trainer(gt, out, self.bank)
            return logits, loss
        else:
            return logits
