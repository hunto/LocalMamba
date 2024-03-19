import sys
import torch

ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip', 'w2', 'w2_flip', 'w7', 'w7_flip')

ckpt = sys.argv[1]

state = torch.load(ckpt, map_location='cpu')['model']

choices = []
for k, v in state.items():
    if 'multi_scan.weights' in k:
        probs = v.view(-1).softmax(-1)
        print(probs)
        topk = probs.topk(4)[1].sort()[0].tolist()
        choices.append(str([ALL_CHOICES[idx] for idx in topk])+',')
for c in choices:
    print(c)
