'''
@Date       : 2023/12/5
@Description: 参考MouseDannce3dDatasetset
20231205: 只用来输出多视角的图片，而没有输出3d gt
'''

import os
import copy
import json
import os.path as osp
import pickle
import tempfile
from collections import OrderedDict
from icecream import ic

import mmcv
import numpy as np
# from icecream import ic
from mmcv import Config
from abc import ABCMeta, abstractmethod
from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.datasets.builder import DATASETS
from torch.utils.data import Dataset
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
# from .kpt_2d_sview_rgb_img_top_down_dataset import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class WholeMouse3dDataset(Dataset, metaclass=ABCMeta):
    ALLOWED_METRICS = {'mpjpe'}
    def __init__(self,
                 ann_file,
                 ann_3d_file,
                 cam_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 bodypart='total',
                 coco_style=True,
                 test_mode=False
                 ):
        """
        :param ann_file: the keypoint annotation file in coco form
        :param cam_file: camera parameter file
        :param img_prefix:
        :param data_cfg:
        :param pipeline:
        :param dataset_info:
        :param bodypart: 'total' or 'face'
        :param coco_style:
        :param test_mode:
        """
        if dataset_info is None:
            cfg = Config.fromfile('configs/datasets/mouse_wholebody.py')
            dataset_info = cfg._cfg_dict['dataset_info']
        self.ann_info = {}
        dataset_info = DatasetInfo(dataset_info)
        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']
        self.ann_info['flip_pairs'] = dataset_info.flip_pairs
        self.ann_info['num_scales'] = 1
        self.ann_info['flip_index'] = dataset_info.flip_index
        # self.ann_info['upper_body_ids'] = dataset_info.upper_body_ids
        # self.ann_info['lower_body_ids'] = dataset_info.lower_body_ids
        self.ann_info['joint_weights'] = dataset_info.joint_weights
        self.ann_info['skeleton'] = dataset_info.skeleton
        self.ann_info['use_different_joint_weights'] = False
        self.sigmas = dataset_info.sigmas
        self.dataset_name = dataset_info.dataset_name

        self.ann_file = ann_file
        self.ann_3d_file = ann_3d_file
        self.img_prefix = img_prefix
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        # self.load_config(data_cfg)
        total_anno = json.load(open(ann_file, 'r'))

        self.bodypart = bodypart
        # load cameras
        # total_calib = json.load(open(cam_file, 'r'))
        self.total_calib = pickle.load(open(cam_file, 'rb'))

        self.groups = self._get_groups(total_anno)

        if coco_style:
            self.coco = COCO(ann_file)
            if 'categories' in self.coco.dataset:
                cats = [
                    cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())
                ]
                self.classes = ['__background__'] + cats
                self.num_classes = len(self.classes)
                self._class_to_ind = dict(
                    zip(self.classes, range(self.num_classes)))
                self._class_to_coco_ind = dict(
                    zip(cats, self.coco.getCatIds()))
                self._coco_ind_to_class_ind = dict(
                    (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                    for cls in self.classes[1:])
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id = self._get_mapping_id_name(
                self.coco.imgs)
        self.db = self._get_db(data_cfg)
        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')


    def _get_cams(self, scene_name):
        calib_data = self.total_calib[f'{scene_name[:3]}']
        # calibrated_cams = [k for k in calib_data.keys() if calib_data[k] is not None]
        # used_cams = sorted(list(set(captured_cams) & set(calibrated_cams)))
        # cam_params = {}
        # for k in used_cams:
        #     cam_params[k] = {}
        #     cam_params[k]['K'] = np.array(calib_data[k]['K']).reshape([3, 3])
        #     cam_params[k]['R'] = np.array(calib_data[k]['R'])
        #     cam_params[k]['T'] = np.array(calib_data[k]['t'])
        # return cam_params
        return calib_data

    def _get_joints_3d(self):
        with open(self.ann_3d_file, 'rb') as f:
            data = json.load(f)
        return data

    def _get_db(self, data_cfg):
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']
        if self.ann_3d_file is not None:
            joints_4d_data = self._get_joints_3d()
        for g, group_item in enumerate(self.groups):
            scene_name = group_item['scene_name']
            # calib_data = self._get_cams(scene_name)
            # cam = list(calib_data.keys())
            db_item = {}
            db_item['data'] = {}
            db_item['scene_name'] = scene_name
            db_item['joints_4d'] = np.array(joints_4d_data[scene_name])[data_cfg['dataset_channel'], :3]
            joints_4d_visible = np.array(joints_4d_data[scene_name])[data_cfg['dataset_channel'], 3:]
            db_item['joints_4d_visible'] = np.minimum(joints_4d_visible, 1.0)
            for i, img_id in enumerate(group_item['image_id']):
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
                obj = self.coco.loadAnns(ann_ids)[0]
                keypoints = np.array(
                    obj['keypoints'] + obj['face_kpts'] +
                    obj['lefthand_kpts'] + obj['leftfoot_kpts'] +
                    obj['righthand_kpts'] + obj['rightfoot_kpts']
                ).reshape(-1, 3)
                keypoints = keypoints[data_cfg['dataset_channel']]
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 1), dtype=np.int32)
                joints_3d[:, :2] = keypoints[:, :2]
                # joints_3d_visible = np.minimum(1, keypoints[:, 2:3] > 0)
                joints_3d_visible = keypoints[:, 2:3]
                # joints_3d_visible = np.maximum(0, keypoints[:, 2:3] - data_cfg['vis_level'])
                image_file = os.path.join(self.img_prefix, self.id2name[img_id])
                db_item['data'][i] = {
                    'image_file': image_file,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'cam_id': obj['cam'],
                    'bbox': obj['bbox'],
                    'rotation': 0,
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                }
                bbox_id = bbox_id + 1
            gt_db.append(db_item)
        return gt_db

    # def _get_db(self, data_cfg):# 都生成24个cam
    #     """use the coco format to index the image, works on train/test dataset"""
    #     gt_db = []
    #     bbox_id = 0
    #     num_joints = self.ann_info['num_joints']
    #     if self.ann_3d_file is not None:
    #         joints_4d_data = self._get_joints_3d()
    #
    #     if self.bodypart == 'total' or self.bodypart == 'skeleton':
    #         for g, group_item in enumerate(self.groups):    # [:2]
    #             db_item = {}
    #             scene_name = group_item['scene_name']
    #             ic(scene_name)
    #             captured_cams = group_item['cams']
    #             db_item['data'] = {}
    #             db_item['scene_name'] = scene_name
    #             db_item['captured_cams'] = captured_cams
    #             db_item['joints_4d'] = np.array(joints_4d_data[g]['kpts_3d'])[data_cfg['dataset_channel'], :]
    #             db_item['joints_4d_visible'] = np.array(joints_4d_data[g]['kpts_3d_vis'])[data_cfg['dataset_channel']]
    #             image_id_iter = iter(group_item['image_id'])
    #             for i in range(24):
    #                 joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
    #                 joints_3d_visible = np.zeros((num_joints, 1), dtype=np.int32)
    #                 if f"cam{i:02d}" in captured_cams:
    #                     img_id = next(image_id_iter)
    #             # for i, img_id in enumerate(group_item['image_id']):
    #                     ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
    #                     obj = self.coco.loadAnns(ann_ids)[0]
    #                     # ic(scene_name, ann_ids, obj['image_id'])    # only for one object
    #                     # joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
    #                     # joints_3d_visible = np.zeros((num_joints, 1), dtype=np.float32)
    #                     keypoints = np.array(
    #                         obj['keypoints'] + obj['face_kpts'] +
    #                         obj['lefthand_kpts'] + obj['leftfoot_kpts'] +
    #                         obj['righthand_kpts'] + obj['rightfoot_kpts']
    #                     ).reshape(-1, 3)
    #                     keypoints = keypoints[data_cfg['dataset_channel']]
    #                     joints_3d[:, :2] = keypoints[:, :2]
    #                     joints_3d_visible = np.minimum(1, keypoints[:, 2:3] > 0)
    #                     image_file = os.path.join(self.img_prefix, self.id2name[img_id])
    #                     # ic(i, image_file)
    #                     db_item['data'][i] = {
    #                         'image_file': image_file,
    #                         'joints_3d': joints_3d,
    #                         'joints_3d_visible': joints_3d_visible,
    #                         'cam_id': obj['cam'],
    #                         'bbox': obj['bbox'],
    #                         'rotation': 0,
    #                         'bbox_score': 1,
    #                         'bbox_id': bbox_id
    #                     }
    #                     bbox_id = bbox_id + 1
    #                 else:
    #                     db_item['data'][i] = {
    #                         'img': np.zeros([2028, 2704, 3]).astype(np.uint8),
    #                         'joints_3d': joints_3d,
    #                         'joints_3d_visible': joints_3d_visible,
    #                         'cam_id': None,
    #                         'bbox': [0.0, 0.0, 10.0, 10.0],
    #                         'rotation': 0,
    #                         'bbox_score': 1,
    #                         'bbox_id': None
    #                     }
    #             gt_db.append(db_item)
    #     return gt_db

    def __getitem__(self, idx):
        """get a sample for coco format"""
        results = copy.deepcopy(self.db[idx])
        cam_params = self._get_cams(results['scene_name'])
        # ic(self.ann_info.keys())
        for key in results['data'].keys():
            if results['data'][key]['cam_id'] is None:
                results['data'][key]['camera_0'] = None
                results['data'][key]['camera'] = None
                results['data'][key]['ann_info'] = self.ann_info
                # results['data'][key]['valid'] = np.array([False])
            else:
                camera_0 = cam_params[f"cam{results['data'][key]['cam_id']}"]
                results['data'][key]['camera_0'] = camera_0
                results['data'][key]['camera'] = copy.deepcopy(camera_0)
                results['data'][key]['ann_info'] = self.ann_info
                # results['data'][key]['valid'] = np.array([True])
        # ic(results.keys())
        # ic(results['data'].keys())
        # ic(results['data'][0].keys())
        return self.pipeline(results)

    def _get_groups(self, total_anno):
        return total_anno['groups']

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db)
        # return len(self.groups)

    def evaluate(self, results, res_folder=None, metric='mpjpe', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}" for human3.6 dataset.'
                    f'Supported metrics are {self.ALLOWED_METRICS}')
        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = []
        for result in results:  # results contain all batches in test set
            preds = result['preds']
            img_metas = result['img_metas']
            batch_size = len(img_metas)
            for i in range(batch_size):  # result in a batch
                kpts.append({
                    'keypoints': preds[i],
                    'joints_4d': img_metas[i]['joints_4d'],
                    'joints_4d_visible': img_metas[i]['joints_4d_visible'][..., 0],
                })
        mmcv.dump(kpts, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(kpts)
            elif _metric == 'p-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='p-mpjpe')
            elif _metric == 'n-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='n-mpjpe')
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return OrderedDict(name_value_tuples)

    def _report_mpjpe(self, keypoint_results, mode='mpjpe'):
        """Cauculate mean per joint position error (MPJPE) or its variants like
        P-MPJPE or N-MPJPE.

        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DH36MDataset.evaluate' for details.
            mode (str): Specify mpjpe variants. Supported options are:

                - ``'mpjpe'``: Standard MPJPE.
                - ``'p-mpjpe'``: MPJPE after aligning prediction to groundtruth
                    via a rigid transformation (scale, rotation and
                    translation).
                - ``'n-mpjpe'``: MPJPE after aligning prediction to groundtruth
                    in scale only.
        """
        preds = []
        gts = []
        masks = []
        for idx, result in enumerate(keypoint_results):
            pred = result['keypoints']
            gt = result['joints_4d']
            mask = result['joints_4d_visible'] > 0
            gts.append(gt)
            preds.append(pred)
            masks.append(mask)
        preds = np.stack(preds)
        gts = np.stack(gts)  # [num_samples, ]
        masks = np.stack(masks) > 0  # [num_samples, num_joints]

        err_name = mode.upper()
        if mode == 'mpjpe':
            alignment = 'none'
        elif mode == 'p-mpjpe':
            alignment = 'procrustes'
        elif mode == 'n-mpjpe':
            alignment = 'scale'
        else:
            raise ValueError(f'Invalid mode: {mode}')
        error = keypoint_mpjpe(preds, gts, masks, alignment)
        ic(error)
        name_value_tuples = [(err_name, error)]
        return name_value_tuples

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.num_joints = data_cfg['num_joints']
        # self.num_cameras = data_cfg['num_cameras']
        self.seq_frame_interval = data_cfg.get('seq_frame_interval', 1)
        self.subset = data_cfg.get('subset', 'train')
        self.need_2d_label = data_cfg.get('need_2d_label', False)
        self.need_camera_param = True
