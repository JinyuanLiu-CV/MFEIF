import cv2
import kornia.utils as utils
import torch
from torch import Tensor


def imread(image_path: str, cuda: bool = True) -> Tensor:
    x = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x = x / 255.0
    x = utils.image_to_tensor(x)
    x = x.type(torch.FloatTensor)
    x = x.cuda() if cuda else x
    return x
