# import IPython
from vat_ladder import main
import json
import os
import argparse

def dict2namespace(dict_):
    p = argparse.Namespace()
    p_dict = vars(p)
    for k, v in dict_.items():
        p_dict[k] = v
    return p


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--id', default=None)
args = parser.parse_args()

if args.cfg is None:
    from config import p
else:
    with open(args.cfg, 'r') as f:
        json_dict = json.load(f)
    p = dict2namespace(json_dict)
    if args.id is not None:
        p.id = args.id

# IPython.embed()

id_seed_dir = p.id + "/" + "seed-{}".format(p.seed) + "/"
save_dir = p.logdir + id_seed_dir
path = save_dir + 'config'
# Write logs to appropriate directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

p_dict = vars(p)
with open(path, 'w') as f:
    json.dump(p_dict, f)

main(p)