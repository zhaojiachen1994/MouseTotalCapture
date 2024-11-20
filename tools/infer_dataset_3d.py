"""use the triangnet to infer the dataset 3d results"""
import json
import os
import warnings
warnings.simplefilter("ignore", UserWarning)
from icecream import ic
from argparse import ArgumentParser
from mmcv import Config
from MTC.datasets import build_dataset
from mmpose.datasets import build_dataloader
from mmpose.apis import init_pose_model
from mmcv.runner import IterLoader
from utils.annos import convert_float32_to_float

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--out-root',
        default='work_dirs/temp',
        help='Root of the output video file. '
             'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    assert args.pose_config is not None
    assert args.pose_checkpoint is not None
    return args


def main():
    args = parse_args()
    config = Config.fromfile(args.pose_config)

    """prepare the output path"""
    out_path = f"{args.out_root}/{args.pose_config.split('/')[-1][:-3]}"
    os.makedirs(out_path, exist_ok=True)

    """initialize the model"""
    model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device.lower())

    """prepare dataset"""
    dataset = build_dataset(config.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False)
    data_loader = IterLoader(data_loader)

    results_3d = {}
    for i in range(5):
    # for i in range(len(data_loader)):
        data = next(data_loader)
        img = data['img'].to(args.device.lower())
        proj_mats = data['proj_mat'].to(args.device.lower())
        img_metas = data['img_metas'].data
        scene = img_metas[0][0]['scene_name']
        res = model.forward(img, proj_mats, img_metas, return_loss=False)
        eval_res = dataset.evaluate([res])
        ic(eval_res)
        results_3d[scene] = res
    results_3d = convert_float32_to_float(results_3d)
    out_3d_file = f"{out_path}/results_3d.json"
    j = json.dumps(results_3d, indent=4)
    with open(out_3d_file, 'w') as f:
        print(j, file=f)


if __name__ == '__main__':
    main()

