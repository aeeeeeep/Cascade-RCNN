# model settings
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
# checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_32xb128_in1k_20220124-eb2d6ada.pth'

model = dict(
    type='CascadeRCNN',

    backbone=dict(
        type='mmcls.ConvNeXt',
        # frozen_stages=1,
        arch='base',
        out_indices=[0, 1, 2, 3],
        # dims=[96, 192, 384, 768],
        drop_path_rate=0.6,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(
        type='FPN',
        # in_channels=[96, 192, 384, 768],
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=5.0),
    ),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,  # 修改类别个数81 类别数量+1
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type="LabelSmoothCrossEntropyLoss", use_sigmoid=False, loss_weight=1.0, label_smooth=0.1),
                # loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type="LabelSmoothCrossEntropyLoss", use_sigmoid=False, loss_weight=1.0, label_smooth=0.1),
                # loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type="LabelSmoothCrossEntropyLoss", use_sigmoid=False, loss_weight=1.0, label_smooth=0.1),
                # loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
        ],)
)

train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type='soft_nms', iou_threshold=0.65),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.65,
                    neg_iou_thr=0.65,
                    min_pos_iou=0.65,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.75,
                    neg_iou_thr=0.75,
                    min_pos_iou=0.75,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ],
        stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            # max_num=1000,
            # nms_thr=0.7,
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.65),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05, nms=dict(type='soft_nms', iou_thr=0.65), max_per_img=100),
        keep_all_stages=False)
# model training and testing settings
# custom_hooks = [dict(type="UnfreezeBackboneEpochBasedHook", unfreeze_epoch=4)]
# dataset settings
dataset_type = 'VOCDataset'
data_root = '/root/autodl-tmp/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Rotate', level=1, max_rotate_angle=90),
    dict(type='Resize', img_scale=[(960, 768),(640, 512)], keep_ratio=True),
    dict(type='Pad', size_divisor=32),
]
train_pipeline = [
    # dict(type='CopyPaste', max_num_pasted=100),
    dict(type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Mosaic', img_scale=(960, 768), pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-(960// 2), - (768// 2))),
    dict(
        type='MixUp',
        img_scale=(960, 768),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(
        type='MixUp',
        img_scale=(960, 768),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file = [data_root + 'VOC2007/ImageSets/Main/trainval.txt'],
            img_prefix = [data_root + 'VOC2007/'],
            pipeline=load_pipeline),
            pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file = data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix = data_root + 'VOC2007/', 
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file = data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999),
                               weight_decay=0.01,
                               paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=4))
# optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=2, grad_clip=dict(max_norm=35, norm_type=4))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 21])
checkpoint_config = dict(interval=4)
# yapf:disable
# log_config = dict(
#     interval=20,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         # dict(type='TensorboardLoggerHook')
#     ])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='Cascade_RCNN_ConvNeXt_B',
                name='ConvNeXt_B'
            )
        )
    ])
# yapf:enable
# runtime settings
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cascade_convnext_b_2'
# load_from = "./data/pretrained/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth"
# load_from = "./data/pretrained/cascade_rcnn_x101_32x4d_fpn_1x_coco_20200316-95c2deb6.pth"
load_from = None
resume_from = None
workflow = [('train', 1)]
