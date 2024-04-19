_base_ = './vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512.py'

model = dict(
    pretrained=None,
    backbone=dict(num_heads=6, embed_dims=192, drop_path_rate=0.1),
    decode_head=dict(num_classes=150, in_channels=[192, 192, 192, 192]),
    neck=None,
    auxiliary_head=dict(num_classes=150, in_channels=192))
