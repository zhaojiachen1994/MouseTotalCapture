from .kpt_2d_sview_rgb_img_top_down_dataset import Kpt2dSviewRgbImgTopDownDataset
from .wholemouse_coco_2d import WholeMouseCocoDataset
from .wholemouse_2d import WholeMouseDataset
from .wholemouse_3d import WholeMouse3dDataset
from .detected_kpt2d_dataset import DetectedKpt2DDataset
from .different_cams import DiffCams3dDataset

__all__ = [
    'WholeMouseDataset', 'WholeMouse3dDataset', "WholeMouseCocoDataset", "DiffCams3dDataset"
]