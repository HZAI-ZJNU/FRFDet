import argparse
import os
from copy import deepcopy

import cv2
import xml.etree.ElementTree as ET

import numpy as np

from ultralytics.utils.plotting import Annotator, colors

CATEGORIES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

CATEGORIES_COLOR = {
    'pedestrian': [200, 200, 200],
    'people': [150, 150, 150],
    'bicycle': [100, 100, 100],
    'car': [128, 0, 128],
    'van': [20, 20, 20],
    'truck': [20, 20, 20],
    'tricycle': [255, 0, 0], 
    'awning-tricycle': [0, 255, 0],
    'bus': [0, 0, 255], 
    'motor': [255, 255, 0], 
}

CATEGORIES_ID = {
    'pedestrian': 0, 'people': 1, 'bicycle': 2, 'car': 3, 'van': 4, 'truck': 5, 'tricycle': 6, 'awning-tricycle': 7,
    'bus': 8, 'motor': 9

    # 'car': 0,
    # 'truck': 1,
    # 'bus': 2


    # 'person': 0,
    # 'bicycle': 1,
    # 'car': 2,
    # 'motorcycle': 3,
    # 'airplane': 4,
    # 'bus': 5,
    # 'train': 6,
    # 'truck' : 7,
    # 'boat': 8,
    # 'traffic light': 9,
    # 'fire hydrant': 10,
    # 'stop sign': 11,
    # 'parking meter': 12,
    # 'bench': 13,
    # 'bird': 14,
    # 'cat': 15,
    # 'dog': 16,
    # 'horse': 17,
    # 'sheep': 18,
    # 'cow': 19,
    # 'elephant': 20,
    # 'bear': 21,
    # 'zebra': 22,
    # 'giraffe': 23,
    # 'backpack': 24,
    # 'umbrella': 25,
    # 'handbag': 26,
    # 'tie': 27,
    # 'suitcase': 28,
    # 'frisbee': 29,
    # 'skis': 30,
    # 'snowboard': 31,
    # 'sports ball': 32,
    # 'kite': 33,
    # 'baseball bat': 34,
    # 'baseball glove': 35,
    # 'skateboard': 36,
    # 'surfboard': 37,
    # 'tennis racket': 38,
    # 'bottle': 39,
    # 'wine glass': 40,
    # 'cup': 41,
    # 'fork': 42,
    # 'knife': 43,
    # 'spoon': 44,
    # 'bowl': 45,
    # 'banana': 46,
    # 'apple': 47,
    # 'sandwich': 48,
    # 'orange': 49,
    # 'broccoli': 50,
    # 'carrot': 51,
    # 'hot dog': 52,
    # 'pizza': 53,
    # 'donut': 54,
    # 'cake': 55,
    # 'chair': 56,
    # 'couch': 57,
    # 'potted plant': 58,
    # 'bed': 59,
    # 'dining table': 60,
    # 'toilet': 61,
    # 'tv': 62,
    # 'laptop': 63,
    # 'mouse': 64,
    # 'remote': 65,
    # 'keyboard': 66,
    # 'cell phone': 67,
    # 'microwave': 68,
    # 'oven': 69,
    # 'toaster': 70,
    # 'sink': 71,
    # 'refrigerator': 72,
    # 'book': 73,
    # 'clock': 74,
    # 'vase': 75,
    # 'scissors': 76,
    # 'teddy bear': 77,
    # 'hair drier': 78,
    # 'toothbrush': 79,
}


def parse_xml(xml_path):
    """
    Parse the XML file to extract bounding box information.
    Args:
        xml_path (str): Path to the XML file.
    Returns:
        list: A list of bounding boxes, each represented as a dictionary.
    """
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


def draw_bboxes(image_path, bboxes, output_path, pil: bool = False):
    """
    Draw bounding boxes on the image and save it.
    Args:
        image_path (str): Path to the input image.
        bboxes (list): List of bounding boxes to draw.
        output_path (str): Path to save the image with drawn bounding boxes.
    """
    image = cv2.imread(image_path)

    annotator = Annotator(
        deepcopy(image),
        None,
        None,
        font=r'TimesNewRoman.ttf',
        pil=pil,  # Classify tasks default to pil=True
    )

    for bbox in bboxes:
        label = bbox["label"]
        xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
        annotator.box_label(
            box=(xmin, ymin, xmax, ymax),
            label=label,
            color=colors(
                CATEGORIES_ID[label],
                True,
            ),
            rotated=False
        )
        # # Draw the rectangle
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), CATEGORIES_COLOR[label], 2)
        #
        # # Add the label text
        # cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (0, 255, 0), 2)

    if pil:
        cv2.imwrite(output_path, np.asarray(annotator.im))
        # annotator.im.save(output_path)
    else:
        cv2.imwrite(output_path, annotator.im)
    print(f"Saved annotated image to {output_path}")


def process_dataset(image_dir, xml_dir, output_dir):
    """
    Process the entire dataset, drawing bounding boxes for each image.
    Args:
        image_dir (str): Directory containing the images.
        xml_dir (str): Directory containing the corresponding XML files.
        output_dir (str): Directory to save the annotated images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        image_name = os.path.splitext(xml_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Image {image_name} not found. Skipping...")
            continue

        bboxes = parse_xml(xml_path)
        draw_bboxes(image_path, bboxes, output_path, pil=True)


def yolo2xml_single(
        txt_path: str,
        xml_path: str,
        img_width: int,
        img_height: int,
        folder_name: str = 'images',
        img_suffix: str = '.bmp'
):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder_name
    ET.SubElement(annotation, 'filename').text = os.path.basename(txt_path).replace('.txt', img_suffix)
    ET.SubElement(annotation, 'path').text = os.path.abspath(txt_path).replace('.txt', img_suffix)

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_width)
    ET.SubElement(size, 'height').text = str(img_height)
    ET.SubElement(size, 'depth').text = '3'

    for line in lines:
        line = line.split()
        print(line)
        class_id, x_center, y_center, width, height = map(float, line)

        x_min = round((x_center - width / 2) * img_width)
        y_min = round((y_center - height / 2) * img_height)
        x_max = round((x_center + width / 2) * img_width)
        y_max = round((y_center + height / 2) * img_height)

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = CATEGORIES[int(class_id)]
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(x_min)
        ET.SubElement(bbox, 'ymin').text = str(y_min)
        ET.SubElement(bbox, 'xmax').text = str(x_max)
        ET.SubElement(bbox, 'ymax').text = str(y_max)

    tree = ET.ElementTree(annotation)
    tree.write(xml_path)


def yolo2xml(
        txt_dir: str,
        xml_dir: str,
        img_dir: str,
        img_suffix: str = '.jpg'
):
    os.makedirs(xml_dir, exist_ok=True)

    txt_names = os.listdir(txt_dir)
    for t_n in txt_names:
        txt_path = os.path.join(txt_dir, t_n)
        img_name = os.path.splitext(t_n)[0] + '.jpg'
        img_path = os.path.join(img_dir, img_name)
        xml_path = os.path.join(xml_dir, os.path.splitext(t_n)[0] + '.xml')
        try:
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            yolo2xml_single(txt_path, xml_path, w, h, t_n, img_suffix)
        except Exception as e:
            print(e, img_path)


class Args(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--image-dir', type=str)
        self.parser.add_argument('--xml-dir', type=str)
        self.parser.add_argument('--output-dir', type=str)

        self.opts = self.parser.parse_args()



if __name__ == '__main__':
    args = Args()
    process_dataset(**vars(args.opts))
