import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import typing as t
from ultralytics import YOLO
from torchvision.transforms import ToTensor
from PIL import Image
import xml.etree.ElementTree as ET

feature_map = None


def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
    global feature_map
    feature_map = output.detach()


def compute_mutual_information(x: torch.Tensor, y: torch.Tensor, bins=32, eps=1e-10):
    # Convert to numpy format, normalize to [0, 255], and quantize to uint8
    x = x.cpu().numpy().astype(np.float32)
    y = y.cpu().numpy().astype(np.float32)

    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Calculate joint histogram and normalize to joint probability
    joint_hist = cv2.calcHist([x, y], [0, 1], None, [bins, 2], [0, 256, 0, 256])
    joint_prob = joint_hist / (np.sum(joint_hist) + eps)  # Add eps to prevent division by zero

    # Calculate marginal distributions
    px = np.sum(joint_prob, axis=1)  # P(X)
    py = np.sum(joint_prob, axis=0)  # P(Y)

    # Avoid log(0) by only operating where pxy > 0
    px_py = np.outer(px, py) + eps
    nonzero_mask = joint_prob > 0

    mi = np.sum(joint_prob[nonzero_mask] * np.log2(joint_prob[nonzero_mask] / px_py[nonzero_mask]))
    return float(mi)


def compute_entropy(region: torch.Tensor, bins: int = 32) -> np.ndarray:
    region = region.cpu().numpy()
    region = cv2.normalize(region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hist = cv2.calcHist([region], [0], None, [bins], [0, 256])
    hist = hist.ravel()
    prob = hist / np.sum(hist) + 1e-9
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def calculate_entropy_from_gt(
        image_path: str,
        gt_boxes: t.List[t.List],  # shape: (N, 4), x1y1x2y2
        model: nn.Module,
        device='cuda' if torch.cuda.is_available() else 'cpu',
):
    feature_maps = []

    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size
    img_resized = image.resize((640, 640))
    img_tensor = ToTensor()(img_resized).unsqueeze(0).to(device)

    _ = model(img_tensor)

    # feat = feature_map.squeeze(0)  # shape: (C, H, W)
    # weights = feat.mean(dim=[1, 2]) # shape: (C)
    # weights = torch.softmax(weights, dim=0)
    # feat_mean = (feat * weights[:, None, None]).sum(dim=0) # (H, W)
    feat = feature_map.squeeze(0)  # shape: (C, H, W)
    feat_mean = feat.mean(dim=0)  # shape: (H, W)

    H_feat, W_feat = feat_mean.shape
    scale_x = W_feat / orig_w
    scale_y = H_feat / orig_h

    # === foreground entropy===
    mask_foreground = torch.zeros_like(feat_mean, dtype=torch.bool)
    foreground_entropies = []
    foreground_mi = []
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = (
            int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y)
        )
        x1, y1, x2, y2 = map(lambda v: max(0, v), [x1, y1, x2, y2])
        region = feat_mean[y1:y2, x1:x2]
        if region.numel() > 0:
            foreground_entropies.append(compute_entropy(region))
            mask_foreground[y1:y2, x1:x2] = True
            # template = torch.ones_like(region)  # label
            # foreground_mi.append(compute_mutual_information(region.flatten(), template.flatten(), bins=16))

    # === background entropy ===
    mask_background = ~mask_foreground
    background_region = feat_mean[mask_background]
    background_entropy = compute_entropy(background_region) if background_region.numel() > 0 else 0

    # if background_region.numel() > 0:
    #     template_bg = torch.zeros_like(background_region)
    #     background_mi = compute_mutual_information(background_region.flatten(), template_bg.flatten(), bins=16)
    # else:
    #     background_mi = 0

    return {
        'foreground_entropy_mean': float(np.mean(foreground_entropies)) if foreground_entropies else 0.0,
        'background_entropy': float(background_entropy),
        # 'foreground_mi': float(np.mean(foreground_mi)) if foreground_mi else 0.0,
        # 'background_mi': float(background_mi),
        'image_path': image_path
    }


def parse_xml(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        label = obj.find("name").text
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        bboxes.append({"label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    return bboxes


def main(
        model_path: str,
        image_dir: str,
        xml_dir: str,
        layer_name: str,
        save_dir: str,
        dataset: str,
        device='cuda' if torch.cuda.is_available() else 'cpu',
):
    model = YOLO(model_path).model
    model = model.eval().to(device)

    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn)
            break
    else:
        raise ValueError(f"Layer {layer_name} not found in model.")
    res = []
    total_b_entropy = 0.
    total_f_entropy = 0.
    total_f_mutual = 0.
    total_b_mutual = 0.
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        image_name = os.path.splitext(xml_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Image {image_name} not found. Skipping...")
            continue
        bboxes: t.List[t.Dict] = parse_xml(xml_path)
        gt_bboxes = [[bbox.get('xmin'), bbox.get('ymin'), bbox.get('xmax'), bbox.get('ymax')] for bbox in bboxes]
        cur_res = calculate_entropy_from_gt(image_path, gt_bboxes, model, device)
        res.append(cur_res)
        total_f_entropy += cur_res['foreground_entropy_mean']
        total_b_entropy += cur_res['background_entropy']
        # total_f_mutual += cur_res['foreground_mi']
        # total_b_mutual += cur_res['background_mi']

    print('avg foreground entropy', total_f_entropy / len(res))
    print('avg background entropy', total_b_entropy / len(res))
    # print('平均前景互信息：', total_f_mutual / len(res))
    # print('平均背景互信息：', total_b_mutual / len(res))

    with open(os.path.join(save_dir, f'{dataset}_vis_entropy.txt'), 'a+', encoding='utf-8') as f:
        f.write(
            f'mean foreground entropy: {total_f_entropy / len(res)} \n'
            f'mean background entropy: {total_b_entropy / len(res)} \n'
            # f'mean foreground mutual: {total_f_mutual / len(res)} \n'
            # f'mean background mutual: {total_b_mutual / len(res)} \n'
            f'{layer_name}\n'
            '============================================================\n\n'
        )
    print(f'successfully process {layer_name}')


if __name__ == '__main__':
    for i in [1, 3, 5, 7, 11, 12, 14, 15, 17, 18, 20, 21]:
        main(
            model_path=r'',
            image_dir=r'',
            xml_dir=r'',
            layer_name=f'model.{i}',
            save_dir=r'',
            dataset='UAVDT',
            device='cuda:0'
        )