import torch
import warnings
import numpy as np
from mmpose.models import builder
from mmpose.models import POSENETS
from .base import BasePose
from PIL import Image
from icecream import ic
import torch.nn as nn
import torch.nn.functional as F
import time
from MTC.models import *
from utils.multiview import *
from utils.visualize import *
try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16

def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


@POSENETS.register_module()
class StrucTriangNet(BasePose):
    def __init__(self,
                 backbone,
                 keypoint_head,
                 triangulate_head=None,
                 attention_head=None,
                 vis_head=None,
                 # score_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
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

        if attention_head is not None:
            self.attention_head = builder.build_head(attention_head)

        if vis_head is not None:
            self.vis_head = builder.build_head(vis_head)

        # if score_head is not None:
        #     self.score_head = builder.build_head(score_head)

        if triangulate_head is not None:
            triangulate_head['train_cfg'] = train_cfg
            triangulate_head['test_cfg'] = test_cfg
            self.triangulate_head = builder.build_head(triangulate_head)

        self.pretrained = pretrained
        self.init_weights()

        # self.conv1 = nn.Conv2d(384, 92, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # self.joint_embeddings = nn.Embedding(92, 64)
        # self.attention = Attention(256, 64, 64)
        # self.out_layers = nn.Sequential(
        #     nn.Linear(68, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 1),
        #     nn.Sigmoid()
        # )




    @property
    def with_keypoint_head(self):
        return hasattr(self, 'keypoint_head')

    @property
    def with_score(self):
        return hasattr(self, 'score_head')


    @property
    def with_triangulate(self):
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

    @staticmethod
    def triangulate_torch(points, proj_mats, confidences=None):
        """
        :param points: (n_cams, n_joints, 2)
        :param proj_mats:
        :param confidences:
        :return:
        """

    @auto_fp16(apply_to=('img',))
    def forward(self, img,
                proj_mat=None,
                img_metas=None,
                target=None,
                target_weight=None,
                joints_3d=None,
                joints_3d_visible=None,
                joints_4d=None,
                joints_4d_visible=None,
                return_loss=True,
                **kwargs
                ):
        if return_loss:
            return self.forward_train(img,
                                      proj_mat,
                                      img_metas,
                                      target,
                                      target_weight,
                                      joints_3d,
                                      joints_3d_visible,
                                      joints_4d,
                                      joints_4d_visible)
        else:
            return self.forward_test(img,
                                     proj_mat,
                                     img_metas,
                                     )

    def forward_train(self, img, proj_mats, img_metas=None,
                      target=None, target_weight=None,
                      joints_3d=None, joints_3d_visible=None,
                      joints_4d=None, joints_4d_visible=None, **kwargs):
        [bs, num_cams, num_channel, h_img, w_img] = img.shape   # bs = 1
        img = img.reshape(-1, *img.shape[2:])   # bs should be 1
        target = target.reshape(-1, *target.shape[2:])
        target_weight = target_weight.reshape(-1, *target_weight.shape[2:])
        proj_mats = proj_mats[0]    # bs = 1

        target_weight = torch.minimum(torch.tensor(1), target_weight)   # 0,1,2 -> 0,1
        joints_3d_visible = torch.maximum(torch.tensor(0), joints_3d_visible-1)
        joints_3d_visible = torch.squeeze(joints_3d_visible, 0).float()

        # joints_3d_visible = np.minimum(1, keypoints[:, 2:3] > 0)

        # start_time = time.time()
        features = self.backbone(img)
        # ic(features.shape)
        heatmap = self.keypoint_head(features)
        feat_3d = self.heatmap_to_3dfeature(heatmap, proj_mats,
                                            img_metas[0]['cams_coord'],
                                            arrow_index=[0, 36, 52, 7]) # [num_cams, 4]
        feat_3d = torch.from_numpy(feat_3d).float().to(heatmap.device)  # [n_img, 4]

        scores = self.vis_head(features, feat_3d)

        # scores = self.attention_head(features, feat_3d) # [num_cams, 92]
        # scores = self.score_head(features)

        # vis_prob = F.softmax(scores, dim=1)#*scores.shape[0]
        # ic(vis_prob)
        # vis_prob = scores[..., 0]
        kpts_3d_pred, _ = self.triangulate_head(heatmap, proj_mats, vis_prob=scores, reproject=False)    # [n_joints, 4] [x, y, z, residual]
        losses = dict()
        if self.with_keypoint_head and target is not None:
            if self.train_cfg.get('use_2d_sup', False):
                keypoint2d_losses = self.keypoint_head.get_loss(heatmap, target, target_weight)
                losses.update(keypoint2d_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(heatmap, target, target_weight)
            losses.update(keypoint_accuracy)

        if self.train_cfg.get('visible_loss', False):
            vis_losses = self.vis_head.get_loss(scores, joints_3d_visible.squeeze(-1))
            losses.update(vis_losses)
            vis_accuracy = self.vis_head.get_accuracy(scores, joints_3d_visible.squeeze(-1))
            losses.update(vis_accuracy)

        # if self.train_cfg.get('visible_loss', False):
        #     vis_losses = self.attention_head.get_loss(scores, joints_3d_visible.squeeze(-1))
        #     losses.update(vis_losses)
        #     vis_accuracy = self.attention_head.get_accuracy(scores, joints_3d_visible.squeeze(-1))
        #     losses.update(vis_accuracy)

        if self.with_triangulate and joints_4d is not None and self.train_cfg.get('use_3d_sup', False):
            sup_3d_loss = self.triangulate_head.get_sup_loss(
                kpts_3d_pred[..., :-1].unsqueeze(0),
                joints_4d, joints_4d_visible)
            losses.update(sup_3d_loss)

        if self.with_triangulate and self.train_cfg.get('use_3d_unsup', False):
            upsup_3d_loss = self.triangulate_head.get_unSup_loss(kpts_3d_pred[..., -1], joints_4d_visible)
            losses.update(upsup_3d_loss)
        return losses

    def forward_test(self, img, proj_mats, img_metas, **kwargs):
        [bs, num_cams, num_channel, h_img, w_img] = img.shape  # bs = 1
        img = img.reshape(-1, *img.shape[2:])  # bs should be 1
        proj_mats = proj_mats[0]  # bs = 1
        features = self.backbone(img)
        heatmap = self.keypoint_head(features)
        feat_3d = self.heatmap_to_3dfeature(heatmap, proj_mats,
                                            img_metas[0][0]['cams_coord'],    # [0]['cams_coord']
                                            arrow_index=[0, 36, 52, 7])
        feat_3d = torch.from_numpy(feat_3d).float().to(heatmap.device)
        # scores = self.attention_head(features, feat_3d)
        scores = self.vis_head(features, feat_3d)

        kpts_3d_pred, kpts_2d_pred = self.triangulate_head(heatmap, proj_mats, vis_prob=scores, reproject=False)  # [n_joints, 4] [x, y, z, residual]
        # dataset_info = "D:\PycharmProjects\MouseTotalCapture\configs\datasets\mouse_wholebody_plot.py"
        # config = mmcv.Config.fromfile(dataset_info)
        # dataset_info = DatasetInfo(config._cfg_dict['dataset_info'])
        # img = imshow_2d(img[0].cpu().numpy().transpose([2, 1, 0])*255,
        #                      [kpts_2d_pred.cpu().numpy()[0]], dataset_info
        #                     )
        # ic(img.shape)
        # plt.imshow(img)


        result = {}
        result['preds_2d'] = kpts_2d_pred[:, :, :, 0].detach().cpu().numpy()   #
        result['preds'] = [kpts_3d_pred[:, :3].detach().cpu().numpy()]
        result['res_triang'] = kpts_3d_pred[:, 3].detach().cpu().numpy()
        result['img_metas'] = img_metas

        return result

    def show_result(self, **kwargs):
        pass

    def heatmap_to_3dfeature(self, heatmap, proj_mats, cams_coord, arrow_index=[]):
        """
        heatmap to 2d keypoint to 3d keypoint to 3d feature
        :param heatmap: 2d heatmap, torch.tensor, [bs, num_joints, h_heatmap, w_heatmap]
        :param cams_coord: [num_cams, 3]
        :return: features [num_cams, 4] 4 dimensional features
        """
        N, J, h_map, w_map = heatmap.shape
        heatmap_np = heatmap.detach().cpu().numpy()
        proj_mats_np = proj_mats.detach().cpu().numpy()
        kpt2d_map, maxvals = _get_max_preds(heatmap_np)     # [N, J, 2], [N, J, 1]
        kpt2d_img = kpt2d_map * 4   # here h_img/h_map = w_img/w_map = 4
        feat_kpts_2d = kpt2d_img[:, arrow_index]
        maxvals = maxvals[:, arrow_index]
        n_cams = len(cams_coord)
        indices = np.argpartition(maxvals, -3, axis=0)[-3:]
        flag = np.zeros_like(maxvals, dtype=bool)
        np.put_along_axis(flag, indices, True, axis=0)

        detected = ((maxvals > 0.9) | flag)*1.0
        # ic(detected)
        kpt2d = np.concatenate([feat_kpts_2d, detected], axis=-1)

        feat_kpts_3d = triangulate_joints(kpt2d, proj_mats_np,
                                   kpt_thr=0.5, use_ransac=True,
                                   ransac_niter=50, ransac_epsilon=10)  # [4, 3]
        cams_coord_rel = cams_coord - feat_kpts_3d[-1]
        cams_coord_rel = cams_coord_rel / (np.linalg.norm(cams_coord_rel, axis=1, keepdims=True))
        v1 = feat_kpts_3d[0] - (feat_kpts_3d[-1] + feat_kpts_3d[0]) / 2
        v2 = feat_kpts_3d[2] - (feat_kpts_3d[1] + feat_kpts_3d[2]) / 2
        v = np.vstack([v1, v2])
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        cos = np.dot(cams_coord_rel, v.T)
        cos_m_z = np.tile(v[0, -1:], (n_cams, 1))
        cos_c_z = cams_coord_rel[:, -1:]
        features = np.hstack([cos, cos_m_z, cos_c_z])
        return features











