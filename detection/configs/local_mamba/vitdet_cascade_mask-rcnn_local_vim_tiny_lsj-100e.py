_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    './lsj-100e_coco-instance.py',
]

custom_imports = dict(imports=['vitdet', 'model'])

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

directions = (
    ['h', 'v_flip', 'w7', 'w7_flip'],
    ['h_flip', 'w2_flip', 'w7', 'w7_flip'],
    ['h', 'h_flip', 'v', 'w7'],
    ['h', 'h_flip', 'v', 'v_flip'],
    ['h', 'h_flip', 'v', 'v_flip'],
    ['h', 'h_flip', 'v', 'v_flip'],
    ['h_flip', 'v', 'v_flip', 'w7'],
    ['h_flip', 'v', 'v_flip', 'w2_flip'],
    ['h', 'h_flip', 'v', 'v_flip'],
    ['h', 'h_flip', 'v', 'v_flip'],
    ['h', 'v', 'v_flip', 'w2'],
    ['h', 'v', 'v_flip', 'w2_flip'],
    ['h', 'h_flip', 'v_flip', 'w7'],
    ['h_flip', 'v', 'v_flip', 'w2'],
    ['h', 'h_flip', 'v', 'v_flip'],
    ['h', 'v', 'w2', 'w2_flip'],
    ['v', 'v_flip', 'w2', 'w7'],
    ['h', 'h_flip', 'v', 'w2'],
    ['h', 'h_flip', 'w2_flip', 'w7'],
    ['v', 'v_flip', 'w2', 'w2_flip'],
)

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='MM_LocalVim',
        pretrained_ckpt='https://github.com/hunto/LocalMamba/releases/download/v1.0.0/local_vim_tiny.ckpt',
        img_size=(1024, 1024),
        drop_path_rate=0.1,
        patch_size=16,
        embed_dim=192,
        depth=20,
        rms_norm=False,
        residual_in_fp32=True,
        fused_add_norm=False,
        final_pool_type='mean',
        if_abs_pos_embed=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2",
        directions=directions,
        out_indices=[19],
    ),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=192,
        in_channels=[48, 96, 192, 192],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

custom_hooks = [dict(type='Fp16CompresssionHook')]
optim_wrapper = dict(
    type="OptimWrapper",
    clip_grad=dict(max_norm=35, norm_type=2))
train_dataloader = dict(batch_size=2)
