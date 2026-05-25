import argparse
import os
import cv2
import typing as t

import numpy as np
from PIL import Image

BLANK_WIDTH = 20


def concat_images(image_paths: t.List[str], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    images = [cv2.imread(path) for path in image_paths]
    height = images[0].shape[0]
    total_width = sum([img.shape[1] for img in images]) + (len(image_paths) - 1) * BLANK_WIDTH

    concat_res = np.full((height, total_width, 3), 255, dtype=np.uint8)

    x_offset = 0
    for img in images:
        concat_res[:, x_offset: x_offset + img.shape[1]] = img
        x_offset += img.shape[1] + BLANK_WIDTH

    cv2.imwrite(os.path.join(save_dir, os.path.basename(image_paths[0])), concat_res)


def main(image_dirs: t.List[str], save_dir: str):
    images_name = os.listdir(image_dirs[0])
    for name in images_name:
        concat_images([os.path.join(d, name) for d in image_dirs], save_dir)
        print(f'successfully process {name}')