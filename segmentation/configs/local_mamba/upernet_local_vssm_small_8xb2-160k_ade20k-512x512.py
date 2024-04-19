_base_ = [
    './upernet_local_vssm_tiny_8xb2-160k_ade20k-512x512.py'
]


directions = [
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v_flip', 'w2', 'w7_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['v', 'v_flip', 'w2_flip', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h_flip', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v_flip', 'w2_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w7_flip'],
        ['v', 'v_flip', 'w7', 'w7_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'w7', 'w7_flip'],
        ['h', 'v_flip', 'w2', 'w2_flip'],
        ['h', 'v_flip', 'w2', 'w7'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v_flip', 'w7', 'w7_flip'],
        ['h', 'v', 'w7', 'w7_flip']
]


model = dict(
    backbone=dict(
        directions=directions,
        pretrained="https://github.com/hunto/LocalMamba/releases/download/v1.0.0/local_vssm_small.ckpt",
        dims=96,
        depths=(2, 2, 27, 2),
        drop_path_rate=0.2,
    ),
)
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(type='AmpOptimWrapper')
