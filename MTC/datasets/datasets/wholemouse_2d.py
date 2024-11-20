import os
import json
from icecream import ic
import os.path as osp
import tempfile
from collections import OrderedDict
import numpy as np
import warnings
from mmcv import Config, deprecated_api_warning
from mmpose.core.post_processing import oks_nms, soft_oks_nms
from mmpose.datasets.builder import DATASETS
from .kpt_2d_sview_rgb_img_top_down_dataset import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class WholeMouseDataset(Kpt2dSviewRgbImgTopDownDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info,
                 bodypart='total',
                 coco_style=True,
                 test_mode=False
                 ):

        # cfg = Config.fromfile(dataset_info)
        # dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            coco_style=coco_style,
            test_mode=test_mode)

        self.bodypart = bodypart
        self.db = self._get_db(data_cfg)
        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self, data_cfg):
        """load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']
        # top_cams = ['00', '01', '02', '03', '04', '05', '06', '07']
        # middle_cams = ['08', '09', '10', '11', '12', '13', '14', '15']
        # bottom_cams = ['16', '17', '18', '19', '20', '21', '22', '23']

        if self.bodypart == 'total':
            for img_id in self.img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
                objs = self.coco.loadAnns(ann_ids)
                for obj in objs:
                    # if max(obj['keypoints']) == 0:
                    #     continue
                    joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                    joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
                    # if obj['cam'] in bottom_cams:
                    keypoints = np.array(
                                    obj['keypoints'] + obj['face_kpts'] +
                                    obj['lefthand_kpts'] + obj['leftfoot_kpts'] +
                                    obj['righthand_kpts'] + obj['rightfoot_kpts']
                                ).reshape(-1, 3)
                    bbox = obj['bbox']
                    keypoints = keypoints[data_cfg['dataset_channel']]
                    joints_3d[:, :2] = keypoints[:, :2]
                    # joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3] > data_cfg['vis_level'])
                    joints_3d_visible[:, :2] = np.maximum(0, keypoints[:, 2:3] - data_cfg['vis_level'])

                    image_file = os.path.join(self.img_prefix, self.id2name[img_id])
                    gt_db.append({
                        'image_file': image_file,
                        'rotation': 0,
                        'joints_3d': joints_3d,
                        'joints_3d_visible': joints_3d_visible,
                        'dataset': self.dataset_name,
                        'bbox': bbox,
                        'bbox_score': 1,
                        'bbox_id': bbox_id
                    })
                    bbox_id = bbox_id + 1

        elif self.bodypart == 'skeleton':
            for img_id in self.img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
                objs = self.coco.loadAnns(ann_ids)
                for obj in objs:
                    if max(obj['keypoints']) == 0:
                        continue
                    joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                    joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
                    # if obj['cam'] in bottom_cams:
                    # ic(np.array(obj['keypoints']).shape)
                    # keypoints = np.array(obj['keypoints']+obj['face_kpts'][12, 28]]).reshape(-1, 3)
                    keypoints = np.concatenate([np.array(obj['keypoints']).reshape(-1, 3),
                                                np.array(obj['face_kpts']).reshape(-1, 3)[[12, 28]]])
                    # keypoints = np.concatenate([np.array(obj['keypoints']), np.array(obj['face_kpts'])[[12, 28]]])
                    # ic(keypoints.shape)


                    bbox = obj['bbox']
                    keypoints = keypoints[data_cfg['dataset_channel']]
                    joints_3d[:, :2] = keypoints[:, :2]
                    joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3] > 0)
                    image_file = os.path.join(self.img_prefix, self.id2name[img_id])
                    gt_db.append({
                        'image_file': image_file,
                        'rotation': 0,
                        'joints_3d': joints_3d,
                        'joints_3d_visible': joints_3d_visible,
                        'dataset': self.dataset_name,
                        'bbox': bbox,
                        'bbox_score': 1,
                        'bbox_id': bbox_id
                    })
                    bbox_id = bbox_id + 1

        elif self.bodypart == 'face':
            for img_id in self.img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
                objs = self.coco.loadAnns(ann_ids)
                for obj in objs:
                    if obj['face_valid']:
                        joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                        joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
                        keypoints = np.array(obj['keypoints'][:6]+obj['face_kpts']).reshape(-1, 3)
                        bbox = obj['face_box']
                        joints_3d[:, :2] = keypoints[:, :2]
                        joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3] > 0)
                        image_file = os.path.join(self.img_prefix, self.id2name[img_id])
                        gt_db.append({
                            'image_file': image_file,
                            'rotation': 0,
                            'joints_3d': joints_3d,
                            'joints_3d_visible': joints_3d_visible,
                            'dataset': self.dataset_name,
                            'bbox': bbox,
                            'bbox_score': 1,
                            'bbox_id': bbox_id
                        })
                        bbox_id = bbox_id + 1

        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mAP', **kwargs):
        """Evaluate coco keypoint results. The pose prediction results will be
                saved in ``${res_folder}/result_keypoints.json``.

                Note:
                    - batch_size: N
                    - num_keypoints: K
                    - heatmap height: H
                    - heatmap width: W

                Args:
                    results (list[dict]): Testing results containing the following
                        items:

                        - preds (np.ndarray[N,K,3]): The first two dimensions are \
                            coordinates, score is the third dimension of the array.
                        - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                            scale[1],area, score]
                        - image_paths (list[str]): For example, ['data/coco/val2017\
                            /000000393226.jpg']
                        - heatmap (np.ndarray[N, K, H, W]): model output heatmap
                        - bbox_id (list(int)).
                    res_folder (str, optional): The folder to save the testing
                        results. If not specified, a temp folder will be created.
                        Default: None.
                    metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

                Returns:
                    dict: Evaluation results for evaluation metric.
                """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'EPE', 'mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = []
        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)
        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value






