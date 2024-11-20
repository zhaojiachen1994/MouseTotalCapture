import torch
import torch.nn as nn
from mmpose.models import HEADS
from mmcv.cnn import (constant_init, normal_init)
from mmpose.models.builder import build_loss
from icecream import ic
BN_MOMENTUM = 0.1


def classification_accuracy(pred, gt, thr=0.5):
    """
    Get the classification accuracy.
    :param pred: [N, 92]
    :param gt: [N, 92]
    :param thr: threshold for prediction label
    :return: float, classification accuracy.
    """
    acc = (((pred - thr) * (gt - thr)) > 0).mean()
    return acc

@HEADS.register_module()
class VisHead(nn.Module):
    def __init__(self, in_channels,
                 n_joints=92,
                 vis_loss=None, thr=0.5,
                 ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )


        self.out_layers = nn.Sequential(
            nn.Linear(132, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.joint_embeddings = nn.Parameter(torch.randn(n_joints, 64))
        self.n_joints = n_joints
        if vis_loss is not None:
            self.loss = build_loss(vis_loss)
        self.thr = thr

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs
        else:
            inputs = inputs[0]
        return inputs

    def init_weights(self):
        """Initialize model weights."""
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.fc_layers.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.001, bias=0)

    def forward(self, feat_2d, feat_3d):
        feat_2d = self._transform_inputs(feat_2d)
        feat_2d = self.conv_layers(feat_2d)
        n_img, n_channels = feat_2d.shape[:2]
        feat_2d = feat_2d.view((n_img, n_channels, -1))
        feat_2d = feat_2d.mean(dim=-1)
        feat_2d = feat_2d.unsqueeze(1).repeat(1, self.n_joints, 1)

        embeddings = self.joint_embeddings.unsqueeze(0).repeat(feat_2d.size(0), 1, 1)
        feat_3d = feat_3d.unsqueeze(1).repeat(1, self.n_joints, 1)

        features = torch.cat([feat_2d, embeddings, feat_3d], dim=-1)
        scores = self.out_layers(features)[:, :, 0]
        return scores


    def get_loss(self, scores, gt):
        """Calculate the visibility classification loss,
        scores: [batch_size, num_keypoints, 1], sigmoid output
        gt: [batch_size, num_keypoints, 1], 0 or 1
        """
        losses = dict()
        losses['bce_loss'] = self.loss(scores, gt)
        return losses

    def get_accuracy(self, scores, gt):
        """
        Calculate accuracy for the visibility classification
        :param pred: [batch_size, num_keypoints, 1], sigmoid output
        :param gt: [batch_size, num_keypoints, 1], 0 or 1
        :return:
        """
        accuracy = dict()
        pred = self.decode(scores)
        acc_vis = classification_accuracy(pred,
                                          gt.detach().cpu().numpy(),
                                          thr=self.thr)
        accuracy['acc_vis'] = float(acc_vis)
        return accuracy

    def decode(self, scores):
        """
        transform the predictive scores to 0 or 1 with the threshold
        :param pred: [batch_size, num_joints]
        :return:
        """
        pred = (scores.detach().cpu().numpy() > self.thr).astype(float)
        return pred