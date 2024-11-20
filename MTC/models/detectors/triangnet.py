import torch
import torch.nn as nn
import warnings
import numpy as np
from icecream import ic
from mmpose.models import builder
from mmpose.models import POSENETS
from .base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16

@POSENETS.register_module()
class TriangNet(BasePose):
    def __init__(self,
                 backbone,
                 keypoint_head,
                 triangulate_head=None,
                 score_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        super().__init__()
        self.fp16_enabled = False
        self.backbone = builder.build_backbone(backbone)
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg
            self.keypoint_head = builder.build_head(keypoint_head)

        if score_head is not None:
            self.score_head = builder.build_head(score_head)

        if triangulate_head is not None:
            triangulate_head['train_cfg'] = train_cfg
            triangulate_head['test_cfg'] = test_cfg
            self.triangulate_head = builder.build_head(triangulate_head)

        self.pretrained = pretrained
        self.init_weights()

    @property
    def with_keypoint_head(self):
        return hasattr(self, 'keypoint_head')

    @property
    def with_score(self):
        return hasattr(self, 'score_head')

    @property
    def with_triangulate_head(self):
        return hasattr(self, 'triangulate_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_keypoint_head:
            self.keypoint_head.init_weights()
        if self.with_score:
            self.score_head.init_weights()

    @auto_fp16(apply_to=('img',))
    def forward(self,
                img,
                proj_mat=None,
                img_metas=None,
                target=None,
                target_weight=None,
                # joints_3d=None,
                # joints_3d_visible=None,
                joints_4d=None,
                joints_4d_visible=None,
                valid=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """
        :param img: Input images, [bs, num_cams, 3, h_img, w_img]
        :param proj_mat: project matrix [bs, num_cams, 3, 4]
        :param img_metas: Information about data augmentation
        :param target: target heatmaps [bs, num_cams, num_joints, H, W]
        :param target_weight: Weights across different joint types. [bs, num_cams, num_joints, 1]
        :param joints_4d: the ground-truth 3d keypoint coordinates. [bs, num_joints, 3]
        :param joints_4d_visible: visibility of the 3d gts. [bs, num_joints, 3]
        :param valid: whether the image available [bs, num_cams, 1]
        :param return_loss: True for forward_train, False for forward_test
        :param return_heatmap: Option to return heatmap.
        :param kwargs:
        :return:
        """
        if return_loss:
            return self.forward_train(img,
                                      proj_mat,
                                      img_metas,
                                      target,
                                      target_weight,
                                      joints_4d,
                                      joints_4d_visible,
                                      valid
                                      )
        else:
            return self.forward_test(img,
                                     proj_mat,
                                     img_metas,
                                     valid,
                                     return_heatmap
                                     )

    def forward_train(self, img, proj_mat=None, img_metas=None,
                      target=None, target_weight=None,
                      joints_4d=None, joints_4d_visible=None, valid=None, **kwargs):
        """
        :param img: Input images, [bs, num_cams, 3, h_img, w_img]
        :param proj_mat: project matrix [bs, num_cams, 3, 4]
        :param img_metas: Information about data augmentation
        :param target: target heatmaps [bs, num_cams, num_joints, H, W]
        :param target_weight: Weights across different joint types. [bs, num_cams, num_joints, 1]
        :param joints_4d: the ground-truth 3d keypoint coordinates. [bs, num_joints, 3]
        :param joints_4d_visible: visibility of the 3d gts. [bs, num_joints, 3]
        :param valid: whether the image available [bs, num_cams, 1]
        :param kwargs:
        :return:
        """
        [bs, num_cams, num_channel, h_img, w_img] = img.shape  # bs = 1
        img = img.reshape(-1, *img.shape[2:])  # bs should be 1
        target = target.reshape(-1, *target.shape[2:])
        target_weight = target_weight.reshape(-1, *target_weight.shape[2:])
        proj_mat = proj_mat[0]  # bs = 1

        target_weight = torch.minimum(torch.tensor(1), target_weight)  # 0,1,2 -> 0,1

        # if valid is not None:
        #     valid = valid.squeeze(-1)
        #     img = img[valid]
        #     target = target[valid]
        #     proj_mat = torch.reshape(proj_mat[valid], [bs, -1, 3, 4])
        #     # ic(img.shape)
        # else:
        #     img = img.reshape(-1, *img.shape[2:])
        #     target = target.reshape(-1, *target.shape[2:])
        # ic(img.shape)
        # target_weight = target_weight.reshape(-1, *target_weight.shape[2:])
        features = self.backbone(img)
        heatmap = self.keypoint_head(features)

        if self.with_score:
            scores = self.score_head(features)
        else:
            scores = torch.ones(*target.shape[:2], dtype=torch.float32, device=target.device)

        kpt_3d_pred, _ = self.triangulate_head(heatmap, proj_mat, scores, reproject=False)

        # if return loss
        losses = dict()
        if self.with_keypoint_head and target is not None:
            if self.train_cfg.get('use_2d_sup', False):
                keypoint2d_losses = self.keypoint_head.get_loss(heatmap, target, target_weight)
                losses.update(keypoint2d_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(heatmap, target, target_weight)
            losses.update(keypoint_accuracy)

        if self.with_triangulate_head and joints_4d is not None and self.train_cfg.get('use_3d_sup', False):
            sup_3d_loss = self.triangulate_head.get_sup_loss(kpt_3d_pred[..., :-1].unsqueeze(0), joints_4d, joints_4d_visible)
            losses.update(sup_3d_loss)

        if self.with_triangulate_head and self.train_cfg.get('use_3d_unsup', False):
            unsup_3d_loss = self.triangulate_head.get_unSup_loss(kpt_3d_pred[..., -1], joints_4d_visible)
            losses.update(unsup_3d_loss)

        return losses

    def forward_test(self, img, proj_mat=None,
                     img_metas=None, valid=None,
                     return_heatmap=None, **kwargs):
        # [bs, num_cams, num_channel, h_img, w_img] = img.shape
        # if valid is not None:
        #     valid = valid.squeeze(-1)
        #     img = img[valid]
        #     proj_mat = torch.reshape(proj_mat[valid], [bs, -1, 3, 4])
        # else:
        #     img = img.reshape(-1, *img.shape[2:])
        # proj_mat = proj_mat[0]  # bs = 1
        [bs, num_cams, num_channel, h_img, w_img] = img.shape  # bs = 1
        img = img.reshape(-1, *img.shape[2:])  # bs should be 1
        proj_mat = proj_mat[0]  # bs = 1


        result = {}
        features = self.backbone(img)
        # heatmap = self.keypoint_head(features)
        output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)
        # ic(output_heatmap.shape)
        if self.with_score:
            scores = self.score_head(features)  # [bs*num_cams, num_joints]
        else:
            scores = torch.ones(*output_heatmap.shape[:2], dtype=torch.float32, device=img.device)

        kpts_3d_pred, kpts_2d_pred = self.triangulate_head(torch.tensor(output_heatmap).to(img.device), proj_mat, scores, reproject=False)  # [n_joints, 4] [x, y, z, residual]
        result = {}
        result['preds_2d'] = kpts_2d_pred[:, :, :, 0].detach().cpu().numpy()  #
        result['preds'] = [kpts_3d_pred[:, :3].detach().cpu().numpy()]
        result['res_triang'] = kpts_3d_pred[:, 3].detach().cpu().numpy()
        result['img_metas'] = img_metas

        return result

    def show_result(self, **kwargs):
        pass






