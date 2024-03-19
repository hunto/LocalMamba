import yaml
import sys
from itertools import product
import json


def gen_network(supernet_fp, subnet, output_fp=''):
    supernet = yaml.safe_load(open(supernet_fp, 'r'))
    network = supernet.copy()
    for layer in network['backbone']:
        _, _, _, t, op, *_ = network['backbone'][layer]
        if isinstance(t, (list, tuple)) or isinstance(op, (list, tuple)):
            if not isinstance(t, (list, tuple)):
                t = [t]
            if not isinstance(op, (list, tuple)):
                op = [op]
            has_id = 'id' in op
            if has_id:
                op.remove('id')
            blocks = list(product(t, op))
            if has_id:
                blocks += [(1, 'id')]
            selected_block = blocks[subnet.pop(0)]
            network['backbone'][layer][3] = selected_block[0]
            network['backbone'][layer][4] = selected_block[1]
    assert len(subnet) == 0
    res = []
    dict_formatter(network, res)
    network = '\n'.join(res) + '\n'
    if output_fp != '':
        open(output_fp, 'w').write(network)
    return network


def dict_formatter(item, res, indent=0):
    if isinstance(item, dict):
        for key in item:
            if not isinstance(item[key], dict):
                res.append(' '*indent + key + ': ' + str(item[key]))
            else:
                res.append(' '*indent + key + ':')
                dict_formatter(item[key], res, indent+4)
    


if __name__ == '__main__':
    print(gen_network(sys.argv[1], list(eval(sys.argv[2])), sys.argv[3]))

