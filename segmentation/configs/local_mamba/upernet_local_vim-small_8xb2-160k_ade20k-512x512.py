_base_ = './upernet_local_vim-tiny_8xb2-160k_ade20k-512x512.py'


model = dict(
    pretrained=None,
    backbone=dict(
        pretrained_ckpt='https://github.com/hunto/LocalMamba/releases/download/v1.0.0/local_vssm_small.ckpt',
        drop_path_rate=0.1,
        embed_dim=384,
    ),
    decode_head=dict(
        in_channels=[384, 384, 384, 384]
    ),
    auxiliary_head=dict(
        in_channels=384
    ),
)