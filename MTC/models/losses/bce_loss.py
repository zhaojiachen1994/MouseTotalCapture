import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from mmpose.models import LOSSES


@LOSSES.register_module()
class MyBCELoss(nn.Module):
    """Binary Cross Entropy loss."""

    def __init__(self, use_mask=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.binary_cross_entropy
        self.use_mask = use_mask
        self.loss_weight = loss_weight

    def forward(self, output, target, mask=None):
        """Forward function.
        Args:
            output (torch.Tensor[bs, n_cams, n_joints]): Output classification.
            target (torch.Tensor[bs, n_cams, n_joints]): Target classification.
            mask (torch.Tensor[bs, n_cams] or torch.Tensor[bs, n_cams]):
                Weights across different labels.
        """
        if self.use_mask:
            ic(output.shape, mask.shape)
            assert mask is not None
            loss = self.criterion(output, target, reduction='none')
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(-1)
                loss = (loss * mask).sum() / (mask.sum() * output.shape[-1])
            else:
                loss = (loss * mask).sum() / mask.sum()
        else:
            loss = self.criterion(output, target, reduction='mean')
        return loss * self.loss_weight

@LOSSES.register_module()
class Kpt3dMSELoss(nn.Module):
    def __init__(self, loss_weight=1.):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, output, target, validity=None):
        """
        :param output: 3d keypoints predictions, [bs, n_joints, 3]
        :param target: 3d keypoints ground truth, [bs, n_joints, 3]
        :param validity: 3d keypoints binary validity, [bs, n_joints, 1], 0 or 1
        :return:
        """
        # ic(torch.isnan(output).any(), torch.isnan(target).any(), torch.isnan(validity).any())
        dim = output.shape[-1]
        output = output[validity[..., 0] > 0]
        target = target[validity[..., 0] > 0]
        loss = torch.mean((output - target) ** 2)*10000

        # loss = torch.nansum((output - target) ** 2 * validity)
        # loss = loss / (dim * max(1, torch.sum(validity).item())) * 1000
        return loss * self.loss_weight

@LOSSES.register_module()
class TRLoss(nn.Module):
    def __init__(self, loss_weight=1.):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, residual, validity=None):
        """
        :param residual: [bs, n_joints, 1]
        :param validity: [bs, n_joints, 1]
        :return:
        """
        # ic(torch.sum(validity), )
        # residual = residual[validity>0]
        loss = torch.nanmean(residual)
        # loss = torch.sum(residual*validity)/max(1, torch.sum(validity).item())
        return loss * self.loss_weight

