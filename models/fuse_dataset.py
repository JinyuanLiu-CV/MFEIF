import os

import torch
import torch.utils.data as data
import torchvision

from tools.imread import imread


def get_all_images(path: str):
    li = []
    for par, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith(('.bmp', '.jpg', '.png')):
                li.append(os.path.join(par, name))
    li.sort()
    return li


class FuseDataset(data.Dataset):
    def __init__(self, image_root: str, image_size: int, cuda: bool = True, label: bool = False):
        super(FuseDataset, self).__init__()

        r_path = image_root
        ir_path = os.path.join(r_path, 'IR')
        vi_path = os.path.join(r_path, 'VI')

        self.ir_list = get_all_images(ir_path)
        self.vi_list = get_all_images(vi_path)

        self.random_crop = torchvision.transforms.RandomCrop(image_size, pad_if_needed=True) if image_size else None
        self.cuda = cuda
        self.label = label

    def __getitem__(self, index: int):
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        ir_name = os.path.basename(ir_path).split('.')[0]
        vi_name = os.path.basename(vi_path).split('.')[0]

        assert ir_name == vi_name

        c = self.cuda
        ir = imread(ir_path, c)
        vi = imread(vi_path, c)

        r = self.random_crop
        t = torch.stack([ir, vi], dim=0)
        t = r(t) if r else t

        ir, vi = t[0], t[1]
        return (ir, vi) if not self.label else (ir, vi, ir_name)

    def __len__(self):
        return len(self.ir_list)
