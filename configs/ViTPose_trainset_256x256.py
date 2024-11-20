_base_ = ['../default_runtime.py',
          '../datasets/mouse_wholebody_3.py'
          ]


total_epochs = 150
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='PCK', save_best='PCK')
optimizer = dict(type='Adam', lr=1e-4,)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup='linear', warmup_iters=500,
                 warmup_ratio=0.1, step=[170, 200])
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])

channel_cfg = dict(
    num_output_channels=92,
    num_joints=92,
    dataset_channel=list(range(92)),
    inference_channel=list(range(92))
)

vis_level = 0   # the visibility level to generate heatmap, 0: manual + project, 1: manual
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
    det_bbox_thr=0.8,
    vis_level=vis_level  # the visibility level to generate heatmap, 0: manual + project, 1: manual
)

load_from = "checkpoints/ViTPose/ViTPose+-S/apt36k.pth"

model = dict(
    type='TopDown',
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
        drop_path_rate=0.1),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=384,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type='GaussianHeatmap',
        modulate_kernel=11,
        use_udp=True)
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.0),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.1, prob=0.3),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible',
            'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.05),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]


test_pipeline = val_pipeline


data_root = "D:/Datasets/MTC dataset"
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=16),
    test_dataloader=dict(samples_per_gpu=16),
    train=dict(
        type='WholeMouseDataset',
        ann_file=f'{data_root}/annotations/train_annotations_2d.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}
    ),
    val=dict(
        type='WholeMouseDataset',
        ann_file=f'{data_root}/annotations/test_annotations_2d.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}
    ),
    test=dict(
        type='WholeMouseDataset',
        ann_file=f'{data_root}/annotations/test_annotations_2d.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}})
)