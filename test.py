import logging.config
import os

import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load snapshots folder
sf = config['test']['snapshots_folder']
rf = config['test']['result_folder']
if not os.path.exists(sf):
    os.makedirs(sf)
if not os.path.exists(rf):
    os.makedirs(rf)

# load extractor network
lpm = config['test']['load_pretrain_model'][0]
pm = os.path.join(sf, 'fuse_ext_{}.pth'.format(str(lpm).zfill(3))) if lpm else None
net_ext = Extractor()
net_ext = nn.DataParallel(net_ext)
net_ext.load_state_dict(torch.load(pm, map_location='cpu')) if pm else None
net_ext = net_ext.cuda() if cuda else net_ext
net_ext.eval()

# load constructor network
lpm = config['test']['load_pretrain_model'][1]
pm = os.path.join(sf, 'fuse_con_{}.pth'.format(str(lpm).zfill(3))) if lpm else None
net_con = Constructor()
net_con = nn.DataParallel(net_con)
net_con.load_state_dict(torch.load(pm, map_location='cpu')) if pm else None
net_con = net_con.cuda() if cuda else net_con
net_con.eval()

# load attention network
lpm = config['test']['load_pretrain_model'][2]
pm = os.path.join(sf, 'fuse_att_{}.pth'.format(str(lpm).zfill(3))) if lpm else None
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

# start test
with torch.no_grad():
    for ir, vi, label in tqdm(loader):

        ir_1, ir_b_1, ir_b_2 = net_ext(ir)
        vi_1, vi_b_1, vi_b_2 = net_ext(vi)

        ir_att = net_att(ir)
        vi_att = net_att(vi)

        fus_1 = ir_1 * ir_att + vi_1 * vi_att
        fus_b_1 = ir_b_1 + vi_b_1
        fus_b_2 = ir_b_2 + vi_b_2

        fus_2 = net_con(fus_1, fus_b_1, fus_b_2)

        p = os.path.join(rf, 'FUS_{}.jpg'.format(label[0]))
        imsave(p, fus_2)
