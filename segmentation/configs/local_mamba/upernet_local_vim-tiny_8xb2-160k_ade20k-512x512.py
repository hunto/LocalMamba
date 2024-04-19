_base_ = '../vit/vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py'

custom_imports = dict(imports=['model'])

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

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='MM_LocalVim',
        pretrained_ckpt='https://github.com/hunto/LocalMamba/releases/download/v1.0.0/local_vim_tiny.ckpt',
        img_size=(512, 512),
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
        out_indices=[4, 9, 14, 19],
    ),
    neck=None,
    decode_head=dict(
        in_channels=[192, 192, 192, 192]
    ),
    auxiliary_head=dict(
        in_channels=192
    ),
)

train_dataloader = dict(batch_size=2)