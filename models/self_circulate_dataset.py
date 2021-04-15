import os

import torch.utils.data as data
import torchvision

from tools.imread import imread


class SelfCirculateDataset(data.Dataset):
    def __init__(self, image_root: str, image_size: int = None, cuda: bool = True, label: bool = False):
        super(SelfCirculateDataset, self).__init__()

        images_list = []
        for par, _, names in os.walk(image_root):
            for name in names:
                if name.lower().endswith(('.bmp', '.jpg', '.png')):
                    images_list.append(os.path.join(par, name))

        self.random_crop = torchvision.transforms.RandomCrop(image_size) if image_size else None
        self.images_list = images_list
        self.cuda = cuda
        self.label = label

    def __getitem__(self, index: int):
        image_path = self.images_list[index]
        folder_path = os.path.dirname(image_path).split('/')[-1]
        image_name = os.path.basename(image_path)
        label_name = os.path.join(folder_path, image_name)

        image = imread(image_path, self.cuda)
        image = self.random_crop(image) if self.random_crop else image

        return image if not self.label else (image, label_name)

    def __len__(self) -> int:
        return len(self.images_list)
