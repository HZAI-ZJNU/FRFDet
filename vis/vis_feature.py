import argparse
import math
import os.path
import random
import shutil

import cv2
import numpy as np
import typing as t
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from ultralytics import YOLO

feature_map: t.Optional[torch.Tensor] = None


def hook_fn(module, input, output):
    global feature_map
    feature_map = output


def feature_visualization(x, stage, save_dir: str, n=32):
    """
    Visualize feature maps of a given model module during inference.

    Args:
        x (torch.Tensor): Features to be visualized.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
        save_dir (Path, optional): Directory to save results. Defaults to Path('runs/detect/exp').
    """
    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:

        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
        if n > 0:
            n = min(n, channels)  # number of plots
        else:
            n = channels
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        blocks = [block.detach() for block in blocks]
        for i in range(n):
            fig, ax = plt.subplots()
            img = blocks[i].squeeze().cpu().numpy()
            ax.imshow(img, cmap='viridis')
            ax.axis('off')

            save_path = os.path.join(save_dir, f'stage{stage}_features_maps_{i + 1}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f'Saving feature map {i + 1}/{n} to {save_path}')


def vis_pic_layer(
        target_layer: nn.Module,
        model: nn.Module,
        image_path: str,
        stage: int,
        save_dir: str,
        n: int = 32,
        show_single: bool = False,
        shape: t.Tuple = (224, 224),
        original_size: bool = False
):
    os.makedirs(save_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    width, height = image.size
    image = transform(image).unsqueeze(0)
    hook = target_layer.register_forward_hook(hook_fn)
    image.requires_grad = True
    model = model.eval()
    model(image)
    hook.remove()

    if not show_single:
        feature_visualization(feature_map, stage, save_dir, n)
    else:
        weights = feature_map.mean(dim=[2, 3])
        weights = torch.softmax(weights, dim=1)
        temp_fea = (feature_map * weights[:, :, None, None])

        if original_size:
            resized_fea = temp_fea.detach().cpu()
            resized_fea = F.interpolate(resized_fea, size=(height, width), mode='bilinear', align_corners=False)
            resized_fea = resized_fea.sum(dim=1).squeeze(0).numpy()
        else:
            resized_fea = temp_fea.sum(dim=1).detach().cpu().squeeze(0).numpy()

        plt.figure()
        plt.imshow(resized_fea, cmap='plasma')
        plt.axis('off')
        save_path = os.path.join(save_dir, f'{os.path.basename(image_path)}-stage{stage}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saving feature map to {save_path}')


def main_imagenet(config: str, checkpoint: str, cls_dir: t.List, save_dir: str, n=16, seed=42):
    random.seed(seed)
    model = YOLO(checkpoint).model

    image_path = r''
    temp_dir = os.path.join(save_dir, os.path.basename(image_path))
    os.makedirs(temp_dir, exist_ok=True)
    target_layer1 = model.model[3]
    target_layer2 = model.model[5]
    target_layer3 = model.model[7]
    target_layer4 = model.model[10]
    targets = [target_layer1, target_layer2, target_layer3, target_layer4]
    for i in range(len(targets)):
        vis_pic_layer(targets[i], model, image_path, i, temp_dir, n, show_single=True)


def main_visdrone(
        dataset_dir: str,
        weight: str,
        save_dir: str,
        num: int = 16,
        show_single: bool = False,
        layer: int = 0,
        shape: t.Tuple = (640, 640),
        original_size: bool = False
):
    os.makedirs(save_dir, exist_ok=True)
    model = YOLO(weight).model

    if os.path.isfile(dataset_dir):
        images_paths = [dataset_dir]
    else:
        images_paths = os.listdir(dataset_dir)
    target_layer = model.model[layer]
    for path in images_paths:
        if not show_single:
            cur_save_dir = os.path.join(save_dir, path.split('.')[0])
        else:
            cur_save_dir = save_dir
        cur_path = os.path.join(dataset_dir, path)
        vis_pic_layer(target_layer, model, cur_path, stage=layer, save_dir=cur_save_dir, n=num, show_single=show_single,
                      shape=shape, original_size=original_size)


class Args(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--weight', type=str)
        self.parser.add_argument('--dataset-dir', type=str)
        self.parser.add_argument('--save-dir', type=str)
        self.parser.add_argument('--num', type=int, default=16, help='feature num')
        self.parser.add_argument('--show-single', action='store_true', help='whether aggregate into one feature map')
        self.parser.add_argument('--shape', type=self.str2tuple, default='640x640')
        self.parser.add_argument('--layer', type=int, help='target layer')
        self.parser.add_argument('--original-size', action='store_true')

        self.opts = self.parser.parse_args()

    @staticmethod
    def str2tuple(v):
        try:
            return tuple(map(int, v.split('x')))
        except:
            raise argparse.ArgumentTypeError("Tuple must be a comma-separated list, e.g., '640x640'.")


if __name__ == '__main__':
    args = Args()
    main_visdrone(**vars(args.opts))
