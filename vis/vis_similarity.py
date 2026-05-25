import argparse
import os
import typing as t
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn

from ultralytics import YOLO

feature_map: t.Optional[torch.Tensor] = None

def hook_fn(module, input, output):
    global feature_map
    feature_map = output.detach()
    

def compute_ssim(feature_map: torch.Tensor, save_path: str, num: int = 64, save: bool = True):
    batch, channel, height, width = feature_map.size()
    feature_map = feature_map.squeeze(dim=0).numpy()
    if num <= 0 or num > channel:
        num = channel
    similarity_matrix = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            similarity_matrix[i, j] += ssim(feature_map[i], feature_map[j], win_size=3,
                                            data_range=max(feature_map[i].max(), feature_map[j].max()) - min(
                                                feature_map[i].min(), feature_map[j].min()),
                                            channel_axis=None)
    if save:
        plt.figure(figsize=(12, 8))
        sns.heatmap(similarity_matrix, annot=False, cmap='viridis', cbar=False)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


def vis_pic_layer(
        target_layer: nn.Module,
        model: nn.Module,
        image_path: str,
        stage: int,
        save_dir: str,
        num: int,
        shape: t.Tuple = (224, 224),
        save: bool = True
):
    if save:
        os.makedirs(save_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    width, height = image.size
    image = transform(image).unsqueeze(0)
    hook = target_layer.register_forward_hook(hook_fn)
    image.requires_grad = True
    model = model.eval()
    model(image)
    hook.remove()

    save_path = os.path.join(save_dir, rf'{os.path.splitext(os.path.basename(image_path))[0]}_{stage}.png')
    compute_ssim(feature_map, save_path, num, save)


def main_visdrone(
        dataset_dir: str,
        weight: str,
        save_dir: str,
        num: int,
        layer: int = 0,
        shape: t.Tuple = (640, 640),
        save: bool = False,
):
    if save:
        os.makedirs(save_dir, exist_ok=True)
    model = YOLO(weight).model

    if os.path.isfile(dataset_dir):
        images_paths = [dataset_dir]
    else:
        images_paths = os.listdir(dataset_dir)

    target_layer = model.model[layer]
    for path in images_paths:
        cur_path = os.path.join(dataset_dir, path)
        vis_pic_layer(target_layer, model, cur_path, stage=layer, save_dir=save_dir, shape=shape, num=num, save=save)
        print(rf'successfully process {cur_path}')


class Args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--weight', type=str)
        self.parser.add_argument('--dataset-dir', type=str)
        self.parser.add_argument('--save-dir', type=str)
        self.parser.add_argument('--shape', type=self.str2tuple, default='640x640')
        self.parser.add_argument('--layer', type=int, help='target layer')
        self.parser.add_argument('--num', type=int)
        self.parser.add_argument('--save', action='store_true')

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
