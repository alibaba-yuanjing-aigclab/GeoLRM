from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LogL1(nn.Module):
    """Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.log(1 + torch.abs(pred - gt)).mean()
        else:
            return torch.log(1 + torch.abs(pred - gt))


class EdgeAwareLogL1(nn.Module):
    """Gradient aware Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation
        self.logl1 = LogL1(implementation="per-pixel")

    def forward(self, pred: Tensor, gt: Tensor, rgb: Tensor, mask: Optional[Tensor]):
        logl1 = self.logl1(pred, gt)

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1] - rgb[..., :, 1:]), -3, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :] - rgb[..., 1:, :]), -3, keepdim=True
        )
        lambda_x = torch.exp(-grad_img_x)
        lambda_y = torch.exp(-grad_img_y)

        loss_x = lambda_x * logl1[..., :, :-1]
        loss_y = lambda_y * logl1[..., :-1, :]

        if self.implementation == "per-pixel":
            if mask is not None:
                loss_x[~mask[..., :, :-1]] = 0
                loss_y[~mask[..., :-1, :]] = 0
            return loss_x[..., :-1, :] + loss_y[..., :, :-1]

        if mask is not None:
            assert mask.shape[:2] == pred.shape[:2]
            loss_x = loss_x[mask[..., :, :-1]]
            loss_y = loss_y[mask[..., :-1, :]]

        if self.implementation == "scalar":
            return loss_x.mean() + loss_y.mean()


class TVLoss(nn.Module):
    """TV loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred):
        """
        Args:
            pred: [batch, C, H, W]

        Returns:
            tv_loss: [batch]
        """
        h_diff = pred[..., :, :-1] - pred[..., :, 1:]
        w_diff = pred[..., :-1, :] - pred[..., 1:, :]
        return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))


def geo_scal_loss(pred, ssc_target, semantic=False):

    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, 0, :, :, :]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # # Remove unknown voxels
    # mask = ssc_target != 255
    # nonempty_target = ssc_target != 0
    # nonempty_target = nonempty_target[mask].float()
    # nonempty_probs = nonempty_probs[mask]
    # empty_probs = empty_probs[mask]

    intersection = (ssc_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / ssc_target.sum()
    spec = ((1 - ssc_target) * (empty_probs)).sum() / (1 - ssc_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )
