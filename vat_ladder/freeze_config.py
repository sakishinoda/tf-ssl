import json
import argparse
from config import p

parser = argparse.ArgumentParser()
parser.add_argument('id', nargs='?', default=None)
args = parser.parse_args()


if args.id is None:
    args.id = p.id

save = args.id + '.cfg.json'


p_dict = vars(p)
with open(save, 'w') as f:
    json.dump(p_dict, f)
