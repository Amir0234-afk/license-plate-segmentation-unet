import os, numpy as np
from PIL import Image
from typing import Tuple

def load_image(path: str, target_size: Tuple[int,int]):
    im = Image.open(path).convert('RGB').resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(im, dtype=np.uint8)

def _mask_from_alpha_grayscale(mask_img: Image.Image, target_size):
    if mask_img.mode in ('RGBA','LA'):
        rgb = mask_img.convert('RGB')
        a = mask_img.getchannel('A')
    else:
        rgb = mask_img.convert('RGB')
        a = None
    rgb = rgb.resize((target_size[1], target_size[0]), Image.NEAREST)
    if a is not None:
        a = a.resize((target_size[1], target_size[0]), Image.NEAREST)
        gray = rgb.convert('L')
        g = np.array(gray, dtype=np.uint8)
        alpha = np.array(a, dtype=np.uint8)
        cls = np.zeros_like(g, dtype=np.uint8)
        cls[(alpha==0)] = 0
        cls[(alpha>0) & (g==255)] = 1
        cls[(alpha>0) & (g<255)] = 2
        return cls
    else:
        gray = rgb.convert('L')
        g = np.array(gray, dtype=np.uint8)
        vals = np.unique(g)
        if set(vals.tolist()).issubset({0,1,2}):
            return g.astype(np.uint8)
        cls = np.zeros_like(g, dtype=np.uint8)
        cls[g==255] = 1
        cls[g==0] = 2
        return cls

def load_mask(path: str, target_size: Tuple[int,int], num_classes: int = 3):
    m = Image.open(path)
    cls = _mask_from_alpha_grayscale(m, target_size)
    oh = np.eye(num_classes, dtype=np.float32)[cls]
    return oh
