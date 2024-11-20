_base_ = [
    '../default_runtime.py',
    '../datasets/mouse_wholebody_3.py'
]


total_epochs = 50
checkpoint_config = dict(interval=10)
evaluation = dict(interval=3, metric='mpjpe', by_epoch=True, save_best='MPJPE')
optimizer = dict(type='Adam', lr=1e-4,)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup='linear', warmup_iters=500,
                 warmup_ratio=0.001, step=[170, 200])
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])

channel_cfg = dict(
    num_output_channels=92,
    num_joints=92,
    dataset_channel=list(range(92)),
    inference_channel=list(range(92))
)

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['num_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
)

load_from = "work_dirs_new/pretrain_2d/ViTPose_trainset_256x256/epoch_100.pth"

model = dict(
    type='StrucTriangNet',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 256),
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
        frozen_stages=10
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=384,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),

    vis_head=dict(
        type='VisHead',
        in_channels=384,
        vis_loss=dict(type='MyBCELoss', loss_weight=0.01),
        thr=0.5
    ),

    triangulate_head=dict(
        type='TriangulateHead',
        img_shape=[256, 256],
        heatmap_shape=[64, 64],
        softmax_heatmap=True,
        loss_3d_sup=dict(type='Kpt3dMSELoss',loss_weight=1.),
        tr_loss=dict(type='TRLoss', loss_weight=1e-6),
        det_conf_thr=0.4
    ),
    train_cfg=dict(
        use_2d_sup=True,
        visible_loss=True,
        use_3d_sup=True,
        use_3d_unsup=True
    ),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type='GaussianHeatmap',
        modulate_kernel=11,
        use_udp=True)
)

train_pipeline = [
    dict(
        type="MultiItemProcessKey",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='SquareBbox'),
            dict(type='CropImage',
                 update_camera=True),
            dict(type='ResizeImage',
                 update_camera=True),
            dict(type='ComputeProjMatric'),
            dict(type='ComputeOneCamCoord'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTarget', sigma=2)],
        keys=['data']
    ),
    dict(
        type='ParseData',
        key_parse=['data']
    ),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=['dataset', 'ann_info']
    ),
    dict(
        type='GroupCams',
        keys=['img', 'target', 'target_weight', 'joints_3d',
              'joints_3d_visible', 'proj_mat', 'cams_coord', 'bbox']#, 'valid'
    ),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight', 'proj_mat',
              'joints_3d', 'joints_3d_visible', 'joints_4d', 'joints_4d_visible'],  #, 'valid'
        meta_keys=['image_file', 'cams_coord', 'bbox_offset', 'resize_ratio', 'scene_name'] #'captured_cams',
    )
]

val_pipeline = [
    dict(
        type="MultiItemProcessKey",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='SquareBbox'),
            dict(type='CropImage',
                 update_camera=True),
            dict(type='ResizeImage',
                 update_camera=True),
            dict(type='ComputeProjMatric'),
            dict(type='ComputeOneCamCoord'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])],
        keys=['data']
    ),
    dict(
        type='ParseData',
        key_parse=['data']
    ),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=['dataset', 'ann_info']
    ),
    dict(
        type='GroupCams',
        keys=['img', 'proj_mat', 'cams_coord', 'bbox']#, 'valid'
    ),
    dict(
        type="Collect",
        keys=['img', 'proj_mat'],
        meta_keys=['image_file', 'joints_4d', 'joints_4d_visible',
                   'cams_coord', 'bbox_offset', 'resize_ratio', 'scene_name']
    )
]

test_pipeline = val_pipeline

data_root = "D:/Datasets/MTC dataset"
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='WholeMouse3dDataset',
        ann_file=f'{data_root}/annotations/train_annotations_2d.json',
        ann_3d_file=f'{data_root}/annotations/train_annotations_3d.json',
        cam_file=f'{data_root}/annotations/calib_data.pkl',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}
    ),
    val=dict(
        type='WholeMouse3dDataset',
        ann_file=f'{data_root}/annotations/test_annotations_2d.json',
        ann_3d_file=f'{data_root}/annotations/test_annotations_3d.json',
        cam_file=f'{data_root}/annotations/calib_data.pkl',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}
    ),
    test=dict(
        type='WholeMouse3dDataset',
        ann_file=f'{data_root}/annotations/test_annotations_2d.json',
        ann_3d_file=f'{data_root}/annotations/test_annotations_3d.json',
        cam_file=f'{data_root}/annotations/calib_data.pkl',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}
    )
)