from torch import nn, Tensor
import torch.nn.functional as F
from monai.networks import one_hot

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

class RobustPolyCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor, weight_loss_map=None) -> Tensor:
        pt = (one_hot(target[:, None, ...], num_classes = input.shape[1]) * F.softmax(input, dim=1)).sum(dim=1)
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        sum_ce = (super().forward(input, target.long()) - (1-pt))
        if weight_loss_map is not None:
            sum_ce = sum_ce * weight_loss_map
        return sum_ce.mean()