import os

import cv2
import kornia.utils as utils
from torch import Tensor


def imsave(image_path: str, image: Tensor):
    p = os.path.dirname(image_path)
    if not os.path.exists(p):
        os.makedirs(p)

    x = image
    x = utils.tensor_to_image(x) * 255.0
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, x)
