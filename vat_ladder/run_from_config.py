from config import p
from vat_ladder import main
import json
import os

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