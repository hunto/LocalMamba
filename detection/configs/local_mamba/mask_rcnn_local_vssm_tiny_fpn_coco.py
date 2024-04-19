_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

custom_imports = dict(imports=['model'])

directions = [
        ['h', 'h_flip', 'w7', 'w7_flip'],
        ['h_flip', 'v_flip', 'w2', 'w2_flip'],
        ['h_flip', 'v_flip', 'w2_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'h_flip', 'v_flip', 'w2_flip'],
        ['h_flip', 'v_flip', 'w2', 'w2_flip'],
        ['h', 'w2_flip', 'w7', 'w7_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v_flip', 'w7', 'w7_flip'],
        ['h_flip', 'v', 'w2', 'w7_flip'],
        ['v', 'v_flip', 'w2', 'w7_flip'],
        ['h', 'h_flip', 'v_flip', 'w2_flip'],
        ['v_flip', 'w2_flip', 'w7', 'w7_flip'],
        ['h_flip', 'v_flip', 'w2_flip', 'w7_flip'],
        ['h_flip', 'v', 'w7', 'w7_flip'],
]

model = dict(
    backbone=dict(
        type='MM_LocalVSSM',
        directions=directions,
        out_indices=(0, 1, 2, 3),
        pretrained="https://github.com/hunto/LocalMamba/releases/download/v1.0.0/local_vssm_tiny.ckpt",
        dims=96,
        depths=(2, 2, 9, 2),
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
    ),
)

train_dataloader = dict(batch_size=2) # as gpus=8

optim_wrapper = dict(type='AmpOptimWrapper')