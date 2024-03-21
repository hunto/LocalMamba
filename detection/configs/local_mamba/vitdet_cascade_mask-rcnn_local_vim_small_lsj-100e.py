_base_ = [
    './vitdet_cascade_mask-rcnn_local_vim_tiny_lsj-100e.py',
]


# model settings
model = dict(
    backbone=dict(
        pretrained_ckpt=None,
        drop_path_rate=0.1,
        embed_dim=384,
    ),
    neck=dict(
        backbone_channel=384,
        in_channels=[96, 192, 384, 384]),
    rpn_head=dict(num_convs=2),
)

train_dataloader = dict(batch_size=1)
