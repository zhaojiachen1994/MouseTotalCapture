# MouseTotalCapture


## Introduction

This is the repository of the paper "Mouse Total Capture: 3D Motion and Expression Capture for the Freely Moving Mouse"

</details><div align=center><img src='figures/system.png' width='800' /></div>

## Dataset

MousePano dataset contains the annotation of whole-body [92 keypoints](https://github.com/zhaojiachen1994/MouseTotalCapture/blob/main/figures/kpt_def2.png) for a freely moving mouse. The annotation covers the trunk, limbs, tail, eyes, ears, fingers, and toes, providing a data foundation for fine-grained mouse behavior analysis. The annotations follow the [COCO format](https://cocodataset.org/#format-data). 

The 2D annotation file contains the following items
```
images{[
  'file_name': str,
  'height': int,
  'width': int,
  'id': int
]}

annotations{
  'image_id': int,
  'id': int
  'cam': str,
  'categoryi_d': 1,
  'segmentation': [],
  'num_keypoints': 92,
  'area': float,
  'iscrowd': 0,
  'bbox': list([x, y, w, h]),
  'keypoints': list([x, y, v] * 24),
  'face_kpts': list([x, y, v] * 32),
  'face_valid': 


}

groups{
  r
}
```



The dataset is available [here](https://docs.google.com/forms/d/e/1FAIpQLSfbl1b3TX9y8WMIHZbruuX0inwC9JfEJg74GxReB2vT4WHHgw/viewform?usp=sf_link)
