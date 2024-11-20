import cv2
import numpy as np
import torch
from PIL import Image
from icecream import ic
# from utils.multiview import triangulate
from mmpose.datasets import PIPELINES, Compose
from collections.abc import Sequence
from mmcv.utils import build_from_cfg


@PIPELINES.register_module()
class DummyTransform:
    def __call__(self, results):
        results['dummy'] = True
        return results


@PIPELINES.register_module()
class BodypartsAugTransform:
    """
    Data augmentation with cut tail transform, keep only body without tail.
    Required key: 'joints_3d', 'joints_3d_visible', and 'ann_info'.

    Modifies key: 'scale' and 'center'.

    """

    def __init__(self, prob_aug=0.4, prob_face=0.6):
        self.prob_aug = prob_aug
        self.prob_face = prob_face

    @staticmethod
    def compute_cs(cfg, selected_joints):
        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)

        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        aspect_ratio = cfg['image_size'][0] / cfg['image_size'][1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * 1.2
        return center, scale

    @staticmethod
    def cut_tail(cfg, joints_3d, joints_3d_visible):
        selected_joints = []
        for joint_id in range(cfg['num_joints']):
            if joints_3d_visible[joint_id][0] > 0:
                if joint_id not in cfg['tail_ids']:
                    selected_joints.append(joints_3d[joint_id])
        return selected_joints
        # if len(selected_joints)<10:
        #     return None, None
        # else:
        #     return compute_cs(cfg, selected_joints)

    @staticmethod
    def select_face(cfg, joints_3d, joints_3d_visible):
        selected_joints = []
        for joint_id in range(cfg['num_joints']):
            if joints_3d_visible[joint_id][0] > 0:
                if joint_id in cfg['face_ids']:
                    selected_joints.append(joints_3d[joint_id])
        return selected_joints
        # if len(selected_joints)<10:
        #     return None, None

    def __call__(self, results):
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']

        if np.random.rand() < self.prob_aug:
            if np.random.rand() < self.prob_face:
                selected_joints = self.select_face(results['ann_info'], joints_3d, joints_3d_visible)
                print("performs BodypartsAugTransform: select the face")
            else:
                selected_joints = self.cut_tail(results['ann_info'], joints_3d, joints_3d_visible)
                print("performs BodypartsAugTransform: cut tail")
            if len(selected_joints) < 15:
                c, s = None, None
            else:
                c, s = self.compute_cs(results['ann_info'], selected_joints)
        else:
            c, s = None, None
        if c is not None and s is not None:
            results['center'] = c
            results['scale'] = s

        return results


@PIPELINES.register_module()
class SquareBbox:
    """Makes square bbox from any bbox by stretching of minimal length side
    bbox is defined by xywh

    Required key: 'bbox', 'ann_info'
    Modified key: 'bbox',

    return bbox by xyxy

    """

    def __init__(self, format='xywh'):
        self.format = format

    def __call__(self, results):
        # ic(results['bbox'])
        bbox = results['bbox']
        # ic(bbox)

        if self.format == 'xyxy':  # the det model output
            left, upper, right, lower = bbox
            width = right - left
            height = lower - upper
        elif self.format == 'xywh':  # the dataset annotation format
            left, upper, width, height = bbox
            right = left + width
            lower = upper + height

        if width > height:
            y_center = (upper + lower) // 2
            upper = y_center - width // 2
            lower = upper + width
            height = width
        else:
            x_center = (left + right) // 2
            left = x_center - height // 2
            right = left + height
            width = height

        results['bbox'] = [left, upper, width, height]
        # ic(results['bbox'])

        return results


@PIPELINES.register_module()
class CropImage:
    """Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros

        Required key: 'bbox', 'img', 'ann_info', 'joint_3d'; 'camera' if update_camera is True
        Modified key: 'img', add 'bbox_offset', 'camera', 'joint_3d'
        Didn't modify: 'bbox'

        Args:
            image numpy array of shape (height, width, 3): input image
            bbox tuple of size 4: input bbox (left, upper, right, lower)
            joint_3d: [num_joints, 3], [x, y, 0]

        Returns:
            cropped_image numpy array of shape (height, width, 3): resulting cropped image
    """

    def __init__(self, update_camera=False, update_gt=True):
        self.update_camera = update_camera
        self.update_gt = update_gt

    def __call__(self, results):
        left = results['bbox'][0]
        upper = results['bbox'][1]
        right = results['bbox'][0] + results['bbox'][2]
        lower = results['bbox'][1] + results['bbox'][3]
        if left == 0 and upper == 0 and right == 0 and lower == 0:
            left, upper, right, lower = 1252, 914, 1452, 1114

        image_pil = Image.fromarray(results['img'])

        image_pil = image_pil.crop([left, upper, right, lower])
        results['img'] = np.asarray(image_pil)

        # update the ground truth keypoint coord

        if self.update_gt:  # and results['valid']
            joint_3d = results['joints_3d']  # 2d indeed
            joint_3d[:, 0] = joint_3d[:, 0] - left
            joint_3d[:, 1] = joint_3d[:, 1] - upper
            results['joints_3d'] = joint_3d

        results['bbox_offset'] = results['bbox'][:2]

        if self.update_camera:  # and results['valid']
            camera = results['camera_0']
            # left, upper, right, lower = results['bbox']
            cx, cy = camera['K'][0, 2], camera['K'][1, 2]

            new_cx = cx - left
            new_cy = cy - upper
            results['camera']['K'][0, 2], results['camera']['K'][1, 2] = new_cx, new_cy

        return results


@PIPELINES.register_module()
class ResizeImage:
    """
    resize the croped box into input image size
    """

    def __init__(self, update_camera=False, update_gt=True):
        self.update_camera = update_camera
        self.update_gt = update_gt

    def __call__(self, results):
        # ic(results['image_file'], results['cam_idx'], results['frame_idx'],results['camera']['K'] )
        img = results['img']
        [height_old, width_old, _] = img.shape

        [new_height, new_width] = results['ann_info']['image_size']
        results['img'] = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # update the ground truth 2d keypoint coord
        if self.update_gt:  # and results['valid']
            results['joints_3d'][:, 0] = results['joints_3d'][:, 0] * (new_width / width_old)
            results['joints_3d'][:, 1] = results['joints_3d'][:, 1] * (new_width / width_old)

        # save the resize ratio
        results['resize_ratio'] = new_width / width_old

        if self.update_camera: #and results['valid']
            camera = results['camera']
            fx, fy, cx, cy = camera['K'][0, 0], camera['K'][1, 1], camera['K'][0, 2], camera['K'][1, 2]
            new_fx = fx * (new_width / width_old)
            new_fy = fy * (new_height / height_old)
            new_cx = cx * (new_width / width_old)
            new_cy = cy * (new_height / height_old)
            results['camera']['K'][0, 0], \
                results['camera']['K'][1, 1], \
                results['camera']['K'][0, 2], \
                results['camera']['K'][1, 2] = new_fx, new_fy, new_cx, new_cy
        # ic(results['image_file'], results['cam_idx'], results['frame_idx'], results['camera']['K'])

        return results


@PIPELINES.register_module()
class MultiItemProcessKey:
    """
        Modified from MultiItemProcess, work on selected keys
        Process each item and merge multi-item results to lists.
    :return:
    """

    def __init__(self, pipeline, keys):
        self.pipeline = Compose(pipeline)
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results_ = {}
            for idx, result in results[key].items():
                # ic(idx, result)
                single_result = self.pipeline(result)
                for k, v in single_result.items():
                    if k in results_:
                        results_[k].append(v)
                    else:
                        results_[k] = [v]
            results[key] = results_
        return results


@PIPELINES.register_module()
class ParseData:
    """针对多视角的图片解析出data的部分"""

    def __init__(self, key_parse):
        self.key_parse = key_parse

    def __call__(self, results):
        results_ = {}
        for key in results.keys():
            if key in self.key_parse:
                for key2 in results[key].keys():
                    results_[key2] = results[key][key2]
            else:
                results_[key] = results[key]
        # ic(results.keys())
        return results_


@PIPELINES.register_module()
class GroupCams:
    """
        将每个视角的信息拼接到一个array/tensor/list里面, for multiview dataset
        Required key: 'img', 'target', 'target_weight'
        Modified key: 'img'
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        # ic(results.keys())
        # ic(results['cam_id'])
        # ic(len(results['cam_params']))
        # ic(len(results['proj_mat']))
        for key in self.keys:
            # ic(key, type(results[key][0]))
            if isinstance(results[key][0], torch.Tensor):
                results[key] = torch.stack(results[key])
            elif isinstance(results[key][0], np.ndarray):
                results[key] = np.stack(results[key])
            elif isinstance(results[key][0], str):
                results[key] = [results[key]]
            if key == 'camera':
                results['cam_params'] = {f"cam{results['cam_id'][i]}": results['camera'][i] for i in
                                         range(len(results['camera']))}

            # elif isinstance(results[key][0], dict):
            #     results[key] = [results[key]
        # ic("-", len(results['cam_params']))
        # ic("-", len(results['proj_mat']))
        return results


@PIPELINES.register_module()
class ComputeProjMatric:
    """
    compute the project matrics for camera based on the camera cameras for one camera
    Required key: 'camera'
    Added key: 'proj_matrics' [3, 4]
    """

    def __call__(self, results):
        # n_cams = len(results['camera'])
        # ic(len(results['camera']))
        # proj_metric = np.zeros([n_cams, 3, 4])
        # ic(results['camera']['K'].shape)
        # ic(results['camera']['R'].shape)
        # ic(results['camera']['T'].shape)
        # ic(np.hstack([results['camera'][0]['R'], np.transpose(results['camera'][0]['T'])]))
        # ic(results['camera']['K'])
        if results['camera'] is None:
            results['proj_mat'] = np.zeros([3, 4], dtype=np.float32)
        else:
            proj_mat = results['camera']['K'].dot(np.hstack([results['camera']['R'],
                                                             results['camera']['T'].reshape([3, 1])]))
            # proj_metric = [params['K'].dot(np.hstack([params['R'], np.transpose(params['T'])])) for params in
            #                results['camera']]
            results['proj_mat'] = proj_mat
            # ic(results['image_file'], results['cam_idx'], results['frame_idx'], results['proj_mat'], results['camera']['K'])
        # results['cam_params'] = results['camera']
        # ic(results['cam_params'])
        return results


@PIPELINES.register_module()
class ComputeProjMatGroup:
    """
    compute the project matrics for cameras based on the params for all cameras
    Required key: 'camera'
    Added key: 'proj_mats' [num_cams, 3, 4]
    """

    def __call__(self, results):
        results['proj_mats'] = []
        for k, params in results['camera'].items():
            proj_mat = params['K'].dot(np.hstack([params['R'], params['T'].reshape([3, 1])]))
            results['proj_mats'].append(proj_mat)
        results['proj_mats'] = np.array(results['proj_mats'])
        # ic("==", results.keys())
        return results


@PIPELINES.register_module()
class ComputeCamsCoord:
    """
    compute the global coordinates for all cameras in a scene
    Required key: 'camera'
    Added key: 'cam_coord' [num_cams, 3]
    """

    def __call__(self, results):
        ic(results.keys())
        ic(results['camera'].keys())
        ic(results['camera']['K'])
        # results['cams_coord'] = [- params['T'] @ params['R'] for k, params in results['camera'].items()]
        results['cams_coord'] = np.array([- params['T'] @ params['R'] for k, params in results['camera'].items()])
        return results

@PIPELINES.register_module()
class ComputeOneCamCoord:
    """
    compute the global coordinate of a camera in a scene
    """
    def __call__(self, results):
        results['cams_coord'] = - results['camera']['T'] @ results['camera']['R']
        return results



@PIPELINES.register_module()
class FillMasks:
    """
    fill the not available cams to build the minibatch
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            temp = np.zeros([24, *results[key].shape[1:]], dtype=float)
            temp[results['cam_masks'] == 1] = results[key]
            results[key] = temp
        results['masks'] = np.minimum(results['bbox_mask'], results['cam_masks'])

        return results


@PIPELINES.register_module()
class ComputeBones:
    def __call__(self, results):
        # ic(results.keys())
        # ic(results['prior'].keys())
        bbox_size = np.linalg.norm(results['bboxes'][:, -2:], axis=-1)
        # prior = results['prior']
        neighbor_idx = np.array(results['anno_info']['neighbors'])
        bone1 = np.linalg.norm(results['kpts_2d'] - results['kpts_2d'][:, neighbor_idx[:, 0], :], axis=-1)
        bone1 = bone1 / bbox_size[:, np.newaxis]
        # bone1 = bone1 + 1e-5
        # bone1_errs = (bone1 - np.array(prior['bone1_mean']))/(np.array(prior['bone1_std'])*5)
        # bone1 = np.nan_to_num(bone1) + 1e-5
        # bone1_errs = np.nan_to_num(bone1_errs, nan=0) #+ 1e-5

        bone2 = np.linalg.norm(results['kpts_2d'] - results['kpts_2d'][:, neighbor_idx[:, 1], :], axis=-1)
        bone2 = bone2 / bbox_size[:, np.newaxis]
        # bone2_errs = (bone2 - np.array(prior['bone2_mean']))/(np.array(prior['bone2_std'])*5)
        # bone2 = np.nan_to_num(bone2) + 1e-5
        # bone2_errs = np.nan_to_num(bone2_errs, nan=0) #+ 1e-5

        dis_swap = np.linalg.norm(results['kpts_2d'] - results['kpts_2d'][:, results['anno_info']['flip_index'], :],
                                  axis=-1)
        dis_swap = dis_swap / bbox_size[:, np.newaxis]
        # dis_swap_err = (dis_swap - np.array(prior['dis_swap_mean']))/(np.array(prior['dis_swap_std'])*3)
        # dis_swap = np.nan_to_num(dis_swap, nan=0) + 1e-5
        # dis_swap_err = np.nan_to_num(dis_swap_err, nan=0) #+ 1e-5
        struc_2d = np.stack([bone1, bone2, dis_swap], axis=2)

        # struc_2d = np.stack([bone1, bone1_errs, bone2, bone2_errs, dis_swap, dis_swap_err], axis=2)
        results['struc_2d'] = struc_2d
        return results


@PIPELINES.register_module()
class CombineFeatures:
    def __call__(self, results):
        vis_prob = results['vis_prob'][:, :, np.newaxis]
        det_conf = results['det_conf'][:, :, np.newaxis]
        struc_3d = np.tile(results['struc_3d'][:, np.newaxis, :], (1, 92, 1))
        # features = np.concatenate([det_conf, vis_prob, results['struc_2d'], struc_3d], axis=-1)
        # features = np.concatenate([results['struc_2d'], struc_3d], axis=-1)
        features = results['struc_2d']
        # features = np.concatenate([det_conf, vis_prob], axis=-1)

        # features = vis_prob
        # features = np.concatenate([results['struc_2d'], struc_3d], axis=-1)
        results['features'] = features
        return results


@PIPELINES.register_module()
class AddHandOutliers:
    """Data augmentation with adding outliers for hand, including flip and random outlier"""

    def __init__(self, sample_prob=0.8, joint_prob=0.2):
        self.sample_prob = sample_prob
        self.joint_prob = joint_prob

    def __call__(self, results):
        np.random.seed(42)
        flip_index = results['anno_info']['flip_index']
        # ic(results.keys())
        # ic(results['vis_prob'].shape)
        # ic(results['anno_info'].keys())
        outlier = np.zeros_like(results['vis_prob'])
        if np.random.rand() <= self.sample_prob:
            kpts_2d = results['kpts_2d'].copy()
            for j in [56, 57, 58, 59, 60, 61, 62, 63]:  #, 74, 75, 76, 77, 78, 79, 80, 81
                for c in np.where((results['det_conf'][:, j] > 0.1) & (results['vis_prob'][:, j] > 0.1))[0]:
                    # for c in range(kpts_2d.shape[0]):
                    # ic(j, c, a)
                    if np.random.rand() <= self.joint_prob:
                        if np.random.rand() <= 0.6:
                            results['kpts_2d'][c, j] = kpts_2d[c, flip_index[j]] + np.random.uniform(-10, 10, 2)
                        else:
                            results['kpts_2d'][c, j] = kpts_2d[c, j] + np.random.uniform(10, 20,
                                                                                         2) * 3 * np.random.choice(
                                [-1, 1], p=[0.5, 0.5])
                        outlier[c, j] = 1
        # ic(np.any(outlier))
        # ic(outlier[:, 56:64])
        # ic(outlier.shape)
        results['outlier'] = outlier
        results['outlier_weight'] = outlier * 0.4 + 0.1
        # ic(outlier.shape, results['vis_gt'].shape)
        # ic(np.any(results['outlier']))
        # results['vis_gt2'] = results['vis_gt']
        # results['vis_gt'] = 1-results['outlier']

        return results


# joint_3d_normalize_param = {
#     'left_eye':
#         {'mean':
#              [[-0.00330903, -0.00355966, 0.00571601],
#               [-0.00330747, -0.00370806, 0.00623596],
#               [-0.00348141, -0.00397997, 0.00664867],
#               [-0.00380838, -0.00432674, 0.00682603],
#               [-0.004143, -0.00464821, 0.00676329],
#               [-0.0041367, -0.00450536, 0.00629409],
#               [-0.0039599, -0.00419221, 0.00593199],
#               [-0.0036614, -0.00384527, 0.00573249]],
#         'std':
#              [[0.00313612, 0.00447112, 0.00560616],
#               [0.0032748, 0.00479368, 0.00589483],
#               [0.00346834, 0.00510966, 0.00628224],
#               [0.00370526, 0.00535976, 0.0066749],
#               [0.0039017, 0.00547606, 0.00700471],
#               [0.0037737, 0.00515164, 0.00672306],
#               [0.00356424, 0.00484204, 0.00634088],
#               [0.00333435, 0.00460916, 0.00595523]]},
#     'left_hand':
#         {'mean':
#              [[0.00091443, 0.00075871, -0.00014678],
#               [0.00152981, 0.00206856, -0.00040235],
#               [0.00023704, 0.00114559, -0.00038511],
#               [0.00048278, 0.00280653, -0.00073618],
#               [-0.00046462, 0.0010948, -0.00055426],
#               [-0.00045917, 0.00269324, -0.00104855],
#               [-0.00099786, 0.00041939, -0.00072329],
#               [-0.00136644, 0.00149611, -0.00128257]],
#          'std':
#              [[0.00127818, 0.0010605, 0.0011749],
#               [0.00181353, 0.00167651, 0.00188182],
#               [0.00118799, 0.00107741, 0.00119032],
#               [0.00181655, 0.00198144, 0.00205498],
#               [0.00113017, 0.00109466, 0.0011673],
#               [0.00178233, 0.00192226, 0.0019887],
#               [0.00113681, 0.00105123, 0.0009489],
#               [0.00162682, 0.00157552, 0.00148611]]},
#     'right_eye':
#         {'mean':
#              [[0.0033547, -0.00343383, 0.00566617],
#               [0.00337256, -0.00360048, 0.00618317],
#               [0.00357092, -0.00389563, 0.00658255],
#               [0.00391478, -0.00423502, 0.00673281],
#               [0.0043153, -0.00454949, 0.00663139],
#               [0.00429034, -0.00437121, 0.00615064],
#               [0.00407057, -0.00407698, 0.00581455],
#               [0.00374993, -0.00373412, 0.00564728]],
#         'std':
#              [[0.00316755, 0.00525402, 0.00513354],
#               [0.00330163, 0.00560544, 0.0054388],
#               [0.00348388, 0.00597208, 0.00583509],
#               [0.00366693, 0.00625049, 0.00621563],
#               [0.00381918, 0.00635684, 0.00647785],
#               [0.00368963, 0.00601055, 0.00616332],
#               [0.00350891, 0.00565112, 0.00576473],
#               [0.00331933, 0.00540393, 0.00540712]]},
#     'right_hand':
#         {'mean':
#              [[-0.00094328, 0.00068179, -0.00018961],
#               [-0.00173527, 0.00186889, -0.00041349],
#               [-0.00029093, 0.00112913, -0.00032171],
#               [-0.00075343, 0.0026555, -0.00059557],
#               [0.00038878, 0.00110256, -0.00044568],
#               [0.00030165, 0.00260231, -0.00078571],
#               [0.0009835, 0.00050863, -0.00063885],
#               [0.00127349, 0.00150265, -0.00103157]],
#         'std': [[0.00085531, 0.00111305, 0.00116805],
#                  [0.00138837, 0.00170997, 0.00190604],
#                  [0.00091689, 0.00126979, 0.00116627],
#                  [0.00173776, 0.00197162, 0.00207922],
#                  [0.00100324, 0.00109918, 0.00109073],
#                  [0.00191817, 0.00183784, 0.001918],
#                  [0.00093623, 0.00113085, 0.00088647],
#                  [0.001663, 0.00165308, 0.00142281]]}}




@PIPELINES.register_module()
class GetJointOffset:
    """
    compute the eyes offset related to nose, fingers offset related to hands.
    1. Translate the tail0 to original point
    2. Rotate the neck to yz-plane, x=0
    3. compute the joint offset related to its root joint.
    """

    def __call__(self, results):
        transformed_points = results['kpts_3d']
        global_offset = results['kpts_3d'][7, :]
        global_offset[2] = 0
        transformed_points = transformed_points - global_offset
        theta = np.pi / 2 - np.arctan2(transformed_points[1, 1], transformed_points[1, 0])
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        transformed_points = np.dot(transformed_points, Rz.T)
        # ic(np.isnan(transformed_points).any())
        offset = transformed_points - transformed_points[np.array(results['anno_info']['parent']), :]
        offset_validity = np.logical_and(results['kpts_3d_validity'],
                                         results['kpts_3d_validity'][np.array(results['anno_info']['parent']), :])
        results['offset_3d'] = offset
        results['offset_validity'] = offset_validity
        results['transformed_points'] = transformed_points
        results['global_offset'] = global_offset
        results['theta'] = theta
        return results

@PIPELINES.register_module()
class NormalizeOffset3d:
    def __init__(self, mean=None, std=None, norm_param_file=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        results['offset_3d'] = (results['offset_3d'] - self.mean) / (self.std+1e-10)
        results['offset_mean'] = self.mean
        results['offset_std'] = self.std
        return results