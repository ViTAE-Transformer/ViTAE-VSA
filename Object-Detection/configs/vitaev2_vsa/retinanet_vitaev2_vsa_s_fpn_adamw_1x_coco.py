_base_ = [
    '../_base_/models/retinanet_vitaev2_vsa_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='ViTAEv2_VSA',
        RC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], 
        NC_tokens_type=['VSA', 'VSA', 'VSA', 'VSA'], 
        embed_dims=[64, 64, 128, 256], 
        token_dims=[64, 128, 256, 512], 
        NC_depth=[2, 2, 8, 2], 
        NC_heads=[2, 4, 8, 16],
        RC_heads=[2, 2, 4, 8],
        NC_group=[1, 32, 64, 128], 
        RC_group=[1, 16, 32, 64],
        cpe=True,
        drop_path_rate=0.15,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[64, 128, 256, 512]))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
find_unused_parameters=True
# do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
