import os
import random
import typing as t

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from mmengine import Config

from mmpretrain import get_model

feature_map: t.Optional[torch.Tensor] = None

IMAGENET_1K = r''


def get_pic_from_each_category() -> t.List:
    cls_path = [os.path.join(IMAGENET_1K, name) for name in os.listdir(IMAGENET_1K)]
    res = []
    for path in cls_path:
        temp = random.sample(os.listdir(path), 5)
        res.extend([os.path.join(path, t) for t in temp])
    return res


def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
    global feature_map
    feature_map = output


def transformer():
    compose = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return compose


def get_field_heatmap(
        model: nn.Module,
        image_paths: t.List,
        k: int,
        layer: t.Optional[nn.Module] = None,
        position: str = 'center',
        x: int = None,
        y: int = None,
) -> t.Any:
    heatmap = torch.zeros([224, 224])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    for path in random.sample(image_paths, k=k):
        img = Image.open(path)
        if len(img.mode) == 1 or img.mode != 'RGB':
            continue
        img = transformer()(img)
        img = img.unsqueeze(0).to('cuda')
        hook = layer.register_forward_hook(hook_fn)
        img.requires_grad = True
        # img.retain_grad()
        model(img)
        hook.remove()

        # use softmax to compute feature
        weights = feature_map.mean(dim=[2, 3])
        weights = torch.softmax(weights, dim=1)
        temp_fea = (feature_map * weights[:, :, None, None]).sum(dim=1).squeeze()
        if position == 'center':
            temp_fea[temp_fea.shape[0] // 2 - 1][temp_fea.shape[1] // 2 - 1].backward()
        elif position == 'leftTop':
            temp_fea[0][0].backward()
        elif position == 'rightBottom':
            temp_fea[-1][-1].backward()
        elif 'random' in position:
            temp_fea[x][y].backward()
        grad = torch.abs(img.grad)
        # (H, W)
        grad = grad.mean(dim=1, keepdim=False).squeeze()
        heatmap = heatmap + grad.cpu().numpy()
        img.grad = None

    # use normalization
    mean = heatmap.mean()
    std = heatmap.std()
    heatmap = (heatmap - mean) / std
    heatmap = np.clip((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()), 0, 1)

    cam = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_PINK)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cam)


def get_heatmap(save_dir: str, model, name: str, k: int, position, x: int = None, y: int = None) -> t.Any:
    os.makedirs(save_dir, exist_ok=True)
    image_paths: t.List = get_pic_from_each_category()
    if 'baseline' in name:
        layer = model.backbone.layer4[0]
    else:
        layer = model.backbone.layer4[0]
    img = get_field_heatmap(model, image_paths, k, layer, position, x, y)
    img.save(rf'{save_dir}\{name}_{position}.png')
    return img

  