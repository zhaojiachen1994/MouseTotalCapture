import os
import copy
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import json
import sys
sys.path.append(".\\")
from MTC import *
from tqdm import tqdm

from argparse import ArgumentParser
import argparse
import cv2
import pickle
import mmcv
from mmcv.parallel import collate, scatter
from icecream import ic
from mmpose.apis import (get_track_id,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_tracking_result)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose, ToTensor
from mmpose.core.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh
from mmcv.visualization import imshow_bboxes
import matplotlib.pyplot as plt
from utils.visualize import *

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-root', type=str, help='Folder of video files')
    parser.add_argument('--video-name', type=str, default=None, help='the video name in the video_root')
    parser.add_argument('--downsample-ratio', type=int, default=1)
    parser.add_argument('--show', action='store_true', default=False, help='whether to show visualizations.')
    parser.add_argument('--save-out-video', action='store_true', default=True)
    parser.add_argument(
        '--save-out-images',
        action='store_true',
        default=False)
    parser.add_argument(
        '--out-root',
        default='work_dirs/temp',
        help='Root of the output video file. '
             'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.5,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.5, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        default=False,
        help='(Deprecated, please use --smooth and --smooth-filter-cfg) '
             'Using One_Euro_Filter for smoothing.')
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply a temporal filter to smooth the pose estimation results. '
             'See also --smooth-filter-cfg.')
    parser.add_argument(
        '--smooth-filter-cfg',
        type=str,
        default='configs/_base_/filters/mouse_smoother.py',
        help='Config file of the filter to smooth the pose estimation '
             'results. See also --smooth.')
    parser.add_argument(
        '--radius',
        type=int,
        default=1,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
             'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
             'information when using multi frames for inference in the pose'
             'estimation stage. Default: False.')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    return args


def _pipeline_gpu_speedup(pipeline, device):
    """Load images to GPU and speed up the data transforms in pipelines.

    Args:
        pipeline: A instance of `Compose`.
        device: A string or torch.device.

    Examples:
        _pipeline_gpu_speedup(test_pipeline, 'cuda:0')
    """

    for t in pipeline.transforms:
        if isinstance(t, ToTensor):
            t.device = device

# 转换 float32 到 float
def convert_float32_to_float(data):
    if isinstance(data, dict):
        return {key: convert_float32_to_float(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return data.astype(float).tolist()
    elif isinstance(data, np.float32):
        return data.item()
    elif isinstance(data, list):
        return [convert_float32_to_float(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_float32_to_float(item) for item in data)
    elif isinstance(data, set):
        return {convert_float32_to_float(item) for item in data}
    else:
        return data

def main():
    args = parse_args()
    print('Initializing model...')

    os.makedirs(args.out_root, exist_ok=True)

    # build the detection model from a config file and a checkpoint file
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())
    dataset = pose_model.cfg.data['test']['type']

    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    # dataset_info = Config.fromfile(dataset_info)._cfg_dict['dataset_info']
    dataset_info = DatasetInfo(dataset_info)

    anno_info = {"image_size": np.array(pose_model.cfg.data_cfg.image_size),
                 "heatmap_size": np.array(pose_model.cfg.data_cfg.heatmap_size),
                 "num_joints": pose_model.cfg.data_cfg.num_joints,
                 "flip_pairs": dataset_info.flip_pairs,
                 "num_scales": 1,
                 "flip_index": dataset_info.flip_index,
                 "joint_weights": dataset_info.joint_weights,
                 "skeleton": dataset_info.skeleton,
                 "use_different_joint_weights": False}

    _test_pipeline = copy.deepcopy(pose_model.cfg.test_pipeline)
    test_pipeline = Compose(_test_pipeline)
    _pipeline_gpu_speedup(test_pipeline, next(pose_model.parameters()).device)



    # get the session name
    session_name_dict = {
        "20230916_0003": "S01", "20230916_0005": "S02",
        "20230916_0007": "S03", "20230923_0003": "S04",
        "20230923_0004": "S05", "20230923_0006": "S06",
        "20230923_0007": "S07", "20230923_0008": "S08",
        "20230923_0009": "S09", "20230923_0010": "S10",
        "20230923_0011": "S11" }
    for k in session_name_dict.keys():
        if k in args.video_root:
            session = session_name_dict[k]

    # get the calib data
    cam_file = "data/calib_data.pkl"
    ic(session)
    calib_data = pickle.load(open(cam_file, 'rb'))[session]
    cams = list(calib_data.keys())

    video_names = []
    if args.video_name is None:
        video_list = [s for s in sorted(os.listdir(args.video_root)) if s.endswith('.mp4')] # 含有某些未标定的相机
        for k in cams:
            for v in video_list:
                if f"{k[-2:]}.mp4" in v:
                    video_names.append(v)
    else:
        video_names = [args.video_name]

    # get the video reader
    input_videos = []
    for i, video_file in enumerate(video_names):
        input_videos.append(mmcv.VideoReader(f"{args.video_root}/{video_file}"))
    num_frames = len(input_videos[0])

    total_results = []
    start = 0
    for f in tqdm(range(num_frames)):

        cur_frames = [video.read() for video in input_videos] # [num_cams, width, height, 3]

        if f % args.downsample_ratio != 0:
            continue
        # print(f"Inferring frame {f:06d}/{num_frames:06}")
        mmdet_results = inference_detector(det_model, cur_frames)   #[[array shape[n_box, 5]], []]
        clear_mmdet_results = []
        for r in mmdet_results:
            if len(r[0]) == 0:
                clear_mmdet_results.append(np.array([1252.0, 914.0, 1452.0, 1114.0, 0.1]))
            else:
                clear_mmdet_results.append(r[0][0])

        # mmdet_results = [r[0][0] for r in mmdet_results]
        # mouse_results = process_mmdet_results(mmdet_results, args.det_cat_id)   # [xyxy]
        # ic(mouse_results)
        bboxes = np.vstack(clear_mmdet_results)   # [num_cams, 5] xyxy+score
        # for jjj in range(len(cur_frames)):
        #     bb = mmdet_results[jjj]
        #     img = imshow_bboxes(cur_frames[jjj], bb, top_k = 1, thickness=3, show=False)
        #     plt.imshow(img)
        #     plt.show()
        bboxes = bbox_xyxy2xywh(bboxes)
        db_item = {}
        db_item['data'] = {}
        # db_item['scene_name'] =
        for i in range(len(cur_frames)):    # 对所有的
            db_item['data'][i] = {'img': cur_frames[i],
                                  'bbox': bboxes[i, :4],
                                  'cam_id': cams[i],
                                  'rotation': 0,
                                  'bbox_score': bboxes[i, -1],
                                  'bbox_id': 0,
                                  'camera_0': calib_data[cams[i]],
                                  'camera': copy.deepcopy(calib_data[cams[i]]),
                                  'ann_info': anno_info
                                  }
        data = test_pipeline(db_item)
        batch_data = [data]
        batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
        batch_data = scatter(batch_data, [args.device.lower()])[0]

        with torch.no_grad():
            pose_result = pose_model(
                img = batch_data['img'],
                proj_mat = batch_data['proj_mat'],
                img_metas = batch_data['img_metas'],
                return_loss = False
            )
        # for jjj in range(len(cur_frames)):
        #     plt.imshow(batch_data['img'].detach().cpu().numpy()[0][jjj].transpose([1,2,0]))
        #     ic(batch_data['img'].detach().cpu().numpy()[0][jjj].transpose([1,2,0]))
        #     bboxes = np.array(pose_result['img_metas'][0]['bbox'])
        #     preds_2d = np.array(pose_result['preds_2d'])
        #     aaa = preds_2d[:, jjj, :]
        #     aaa = aaa * bboxes[jjj, 2]/256.0 + bboxes[jjj, :2]
        #
        #     img = imshow_2d(cur_frames[jjj], [aaa], skeleton=dataset_info.skeleton, pose_kpt_color=dataset_info.pose_kpt_color, pose_link_color=dataset_info.pose_link_color, radius=10, thickness=2)
        #     plt.imshow(img)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax = plot_mouse_mpl(ax, np.array(pose_result['preds']))
        # plt.show()

        total_results.append({'preds': pose_result['preds'],
                              'res_triang': pose_result['res_triang']
                              })
        if (f+1) % 500 == 0:
            total_results = convert_float32_to_float(total_results)
            j = json.dumps(total_results, indent=4)
            with open(f"{args.out_root}/results_3d_{start:08d}-{f:08d}.json", 'w') as file:
                print(j, file=file)
            start = f+1
            total_results = []

    if len(total_results) > 0:
        total_results = convert_float32_to_float(total_results)
        j = json.dumps(total_results, indent=4)
        with open(f"{args.out_root}/results_3d_{start:08d}-{num_frames:08d}.json", 'w') as f:
            print(j, file=f)

















if __name__ == '__main__':
    main()