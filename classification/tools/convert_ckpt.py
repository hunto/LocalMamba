import sys
import torch


ori_path = sys.argv[1]
new_path = sys.argv[2]
key = sys.argv[3]

ckpt = torch.load(ori_path, map_location='cpu')
new_ckpt = {}

new_ckpt['state_dict'] = ckpt[key]

torch.save(new_ckpt, new_path)
