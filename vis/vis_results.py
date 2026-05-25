import argparse
import os

import cv2
import torch
import typing as t
from ultralytics import YOLO


def main(weight: str, img_dir: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model = YOLO(weight)
    img_names = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, img_name) for img_name in img_names]
    for p in img_paths:
        res = model(p)
        img = res[0].plot(font=r'TimesNewRoman.ttf', pil=True)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(p)), img)
        print(rf'successfully process {p}')


class Args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--weight', type=str, help='file path of model weight')
        self.parser.add_argument('--img-dir', type=str, help='image folder')
        self.parser.add_argument('--save-dir', type=str, help='the directory utilized for saving output files')
        self.opts = self.parser.parse_args()


if __name__ == '__main__':
    args = Args()
    main(**vars(args.opts))
