import logging.config
import os
import statistics
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from functions.feather_fuse import FeatherFuse
from models.attention import Attention
from models.constructor import Constructor
from models.extractor import Extractor
from models.fuse_dataset import FuseDataset
# load config
from tools.imsave import imsave

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# load logger
lc = config['environment']['log_config']
logging.config.fileConfig(lc)
logs = logging.getLogger()

# load device config
cuda = config['environment']['cuda']
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load snapshots folder
sf = config['test']['snapshots_folder']
rf = config['test']['result_folder']
if not os.path.exists(sf):
    os.makedirs(sf)
if not os.path.exists(rf):
    os.makedirs(rf)

# load extractor network
lpm = config['test']['load_pretrain_model'][0]
pm = os.path.join(sf, 'epoch_ext_{}.pth'.format(str(lpm).zfill(3))) if lpm else None
net_ext = Extractor()
net_ext = nn.DataParallel(net_ext)
net_ext.load_state_dict(torch.load(pm, map_location='cpu')) if pm else None
net_ext = net_ext.cuda() if cuda else net_ext
net_ext.eval()

# load constructor network
lpm = config['test']['load_pretrain_model'][1]
pm = os.path.join(sf, 'epoch_con_{}.pth'.format(str(lpm).zfill(3))) if lpm else None
net_con = Constructor()
net_con = nn.DataParallel(net_con)
net_con.load_state_dict(torch.load(pm, map_location='cpu')) if pm else None
net_con = net_con.cuda() if cuda else net_con
net_con.eval()

# load attention network
lpm = config['test']['load_pretrain_model'][2]
pm = os.path.join(sf, 'epoch_att_{}.pth'.format(str(lpm).zfill(3))) if lpm else None
net_att = Attention()
net_att = nn.DataParallel(net_att)
net_att.load_state_dict(torch.load(pm, map_location='cpu')) if pm else None
net_att = net_att.cuda() if cuda else net_att
net_att.eval()

# load dataloader
iz = config['test']['image_size']
it = config['test']['image_root']
data = FuseDataset(it, iz, cuda, True)
loader = DataLoader(data, 1, False)

# softmax
sm = nn.Softmax(dim=1)

# load feather fuse
ff = FeatherFuse()

# start test
tr = []
with torch.no_grad():
    for ir, vi, label in tqdm(loader):
        st = time.time()

        ir_1, ir_b_1, ir_b_2 = net_ext(ir)
        vi_1, vi_b_1, vi_b_2 = net_ext(vi)

        ir_att = net_att(ir)
        vi_att = net_att(vi)

        fus_1 = ir_1 * ir_att + vi_1 * vi_att
        fus_1 = sm(fus_1)
        fus_b_1, fus_b_2 = ff((ir_b_1, ir_b_2), (vi_b_1, vi_b_2))

        fus_2 = net_con(fus_1, fus_b_1, fus_b_2)

        torch.cuda.synchronize() if cuda else None
        et = time.time()
        tr.append(et - st)

        p = os.path.join(rf, '{}.jpg'.format(label[0]))
        imsave(p, fus_2)

# time record
s = statistics.stdev(tr[1:])
m = statistics.mean(tr[1:])
print('std time: {} \t mean time: {}'.format(s, m))
