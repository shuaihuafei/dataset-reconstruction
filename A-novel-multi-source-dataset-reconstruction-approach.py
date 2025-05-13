import os
import random
import cv2
import numpy as np
import albumentations as A
from albumentations.augmentations.geometric.rotate import Rotate
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
import shutil

random.seed(0)
np.random.seed(0)

class Config:
    # path-background
    BACKGROUND_IMG_DIR = '/xxx/BACKGROUND/JPEGImages'
    BACKGROUND_ANNOTATION_DIR = '/xxx/BACKGROUND/Annotations'
    # path-embody
    EMBEDDED_IMG_DIR = '/xxx/EMBEDDED/JPEGImages'
    EMBEDDED_ANNOTATION_DIR = '/xxx/EMBEDDED/Annotations'
    EMBEDDED_MASK_DIR = '/xxx/EMBEDDED/mask'
    # path-foreground
    OUTPUT_IMG_DIR = '/xxx/AUG/JPEGImages'
    OUTPUT_ANNOTATION_DIR = '/xxx/AUG/Annotations'
    Y_RATIO_RANGE = (0.5, 0.8)
    NUM_EMBEDDED_IMGS_RANGE = [(0, 1, 2, 3),(2, 3, 0, 0)]
    LEFT_LIMIT_BASE = -0.98
    RIGHT_LIMIT_BASE = -0.70
    MIN_GAP = 0.15

class GaussianRotate(A.Rotate):
    def __init__(self, mean=0, std=5, always_apply=False, p=1.0):
        super(GaussianRotate, self).__init__(
            limit=(-15, 15),
            always_apply=always_apply,
            p=p
        )
        self.mean = mean
        self.std = std
        
    def get_params(self):
        angle = np.random.normal(self.mean, self.std)
        angle = max(min(angle, 15.0), -15.0)
        return {"angle": angle}
    
def calculatePositionOffset(
    background_img_shape: Tuple[int, int],
    embedded_img_shape: Tuple[int, int],
    y_ratio: float,
    bbox_y_max: int,
    last_x_offset_set: Optional[List[int]] = None
) -> Tuple[int, int, List[int]]:
    bg_height, bg_width = background_img_shape[:2]
    emb_height, emb_width = embedded_img_shape[:2]

    y_offset = int(bg_height * y_ratio)
    y_offset = max(y_offset, bbox_y_max - int(emb_height/2)) 
    y_offset = min(y_offset, bg_height - emb_height)

    max_x_offset = max(0, bg_width - emb_width)
    if len(last_x_offset_set) != 0:
        while True:
            x_offset = random.randint(0, max_x_offset)
            if all(abs(x_offset - prev_x) >= bg_width * Config.MIN_GAP for prev_x in last_x_offset_set):
                last_x_offset_set.append(x_offset)
                break
    else:
        x_offset = random.randint(0, max_x_offset)
        last_x_offset_set.append(x_offset)

    return y_offset, x_offset

def calculateDynamicScaleLimit(y_ratio: float, y_range: Tuple[float, float]) -> Tuple[float, float]:
    left_ratio, right_ratio = y_range

    left_limit_base = Config.LEFT_LIMIT_BASE
    right_limit_base = Config.RIGHT_LIMIT_BASE

    normalized_ratio = (y_ratio - left_ratio) / (right_ratio - left_ratio)

    left_limit = left_limit_base + (right_limit_base - left_limit_base) * normalized_ratio
    right_limit = left_limit_base + (right_limit_base - left_limit_base) * normalized_ratio

    return left_limit, right_limit

def get_random_files(file_dir: str, extension: str, num_files: int) -> List[str]:
    files = [f for f in os.listdir(file_dir) if f.endswith(extension)]
    if not files:
        return []
    return [os.path.join(file_dir, f) for f in random.sample(files, min(num_files, len(files)))]

def parse_annotation_background(annotation_path: str) -> Optional[List[List[int]]]:
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None, None

    bounding_boxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        coords = [
            int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text),
        ]
        bounding_boxes.append(coords)

    return bounding_boxes

def parse_annotation(annotation_path: str, scale_factor: float) -> Tuple[Optional[List[List[int]]], Optional[List[str]]]:
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None, None

    object_names = [obj.find('name').text for obj in root.findall('object')]
    if not all(name == "USV" for name in object_names):
        print("Not all <name> elements are 'USV'.")
        return None, None

    bounding_boxes, labels = [], []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        coords = [
            int(int(bndbox.find('xmin').text) * scale_factor),
            int(int(bndbox.find('ymin').text) * scale_factor),
            int(int(bndbox.find('xmax').text) * scale_factor),
            int(int(bndbox.find('ymax').text) * scale_factor),
        ]
        bounding_boxes.append(coords)
        labels.append('USV')

    return bounding_boxes, labels

def get_max_ymax(bounding_boxes: List[List[int]]) -> Optional[int]:
    if not bounding_boxes:
        return None
    
    max_ymax = max(box[3] for box in bounding_boxes)
    return max_ymax

def resize_image(image: np.ndarray, target_width: int, target_height: int) -> Tuple[np.ndarray, float]:
    scale_x = target_width / image.shape[1]
    scale_y = target_height / image.shape[0]
    scale_factor = min(scale_x, scale_y)

    resized_width = int(image.shape[1] * scale_factor)
    resized_height = int(image.shape[0] * scale_factor)

    resized_img = cv2.resize(image, (resized_width, resized_height))
    return resized_img, scale_factor

def apply_offsets_to_bboxes(bboxes: List[List[int]], x_offset: int, y_offset: int) -> List[Tuple[int, int, int, int]]:
    return [
        (xmin + x_offset, ymin + y_offset, xmax + x_offset, ymax + y_offset)
        for xmin, ymin, xmax, ymax in bboxes
    ]

def update_annotation_with_objects(annotation_path: str, output_annotation_path: str,
                                    bounding_boxes: List[Tuple[int, int, int, int]],
                                    labels: List[str]) -> None:
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return

    for bbox, label in zip(bounding_boxes, labels):
        obj_element = ET.Element("object")

        ET.SubElement(obj_element, "name").text = label
        ET.SubElement(obj_element, "pose").text = "Unspecified"
        ET.SubElement(obj_element, "truncated").text = "0"

        bndbox_element = ET.SubElement(obj_element, "bndbox")
        ET.SubElement(bndbox_element, "xmin").text = str(int(bbox[0]))
        ET.SubElement(bndbox_element, "ymin").text = str(int(bbox[1]))
        ET.SubElement(bndbox_element, "xmax").text = str(int(bbox[2]))
        ET.SubElement(bndbox_element, "ymax").text = str(int(bbox[3]))

        root.append(obj_element)

    tree.write(output_annotation_path, encoding="utf-8", xml_declaration=True)

def main():
    config = Config()

    background_images = [
        os.path.join(config.BACKGROUND_IMG_DIR, f)
        for f in os.listdir(config.BACKGROUND_IMG_DIR)
        if f.endswith('.jpg')
    ]

    if not background_images:
        print("Error: No background images found.")
        return

    for background_img_path in background_images:
        background_annotation_path = os.path.join(
            config.BACKGROUND_ANNOTATION_DIR, os.path.splitext(os.path.basename(background_img_path))[0] + '.xml'
        )
        bounding_boxes_background = parse_annotation_background(background_annotation_path)
        bbox_y_max = get_max_ymax(bounding_boxes_background)
        if(bbox_y_max is None):
            print(background_annotation_path)

        num_embedded_imgs = random.choices(config.NUM_EMBEDDED_IMGS_RANGE[0], config.NUM_EMBEDDED_IMGS_RANGE[1])[0]
        if num_embedded_imgs == 0:
            print(f"Copy images {background_img_path}.")
            output_img_path = os.path.join(config.OUTPUT_IMG_DIR, os.path.basename(background_img_path))
            output_annotation_path = os.path.join(config.OUTPUT_ANNOTATION_DIR, os.path.basename(background_annotation_path))
            shutil.copy(background_img_path, output_img_path)
            shutil.copy(background_annotation_path, output_annotation_path)
            continue
        embedded_img_paths = get_random_files(config.EMBEDDED_IMG_DIR, '.jpg', num_embedded_imgs)

        if not embedded_img_paths:
            print(f"Error: Could not find embedded images.")
            continue

        background_img = cv2.imread(background_img_path)
        if background_img is None:
            print(f"Error: Could not load background image {background_img_path}.")
            continue

        all_offset_bboxes = []
        all_labels = []

        last_x_offset_set = []
        for embedded_img_path in embedded_img_paths:
            embedded_annotation_path = os.path.join(
                config.EMBEDDED_ANNOTATION_DIR, os.path.splitext(os.path.basename(embedded_img_path))[0] + '.xml'
            )
            embedded_mask_path = os.path.join(
                config.EMBEDDED_MASK_DIR, 'mask_' + os.path.splitext(os.path.basename(embedded_img_path))[0] + '.png'
            )
            embedded_img = cv2.imread(embedded_img_path)
            mask_img = cv2.imread(embedded_mask_path)
            if (embedded_img is None) or (mask_img is None):
                print(f"Error: Could not load embedded image {embedded_img_path}.")
                continue

            resized_embedded_img, scale_factor = resize_image(
                embedded_img, background_img.shape[1], background_img.shape[0]
            )
            resized_mask, _ = resize_image(
                mask_img, background_img.shape[1], background_img.shape[0]
            )

            bounding_boxes, labels = parse_annotation(embedded_annotation_path, scale_factor)
            if bounding_boxes is None:
                continue

            y_ratio = random.uniform(*config.Y_RATIO_RANGE)
            scale_limit = calculateDynamicScaleLimit(y_ratio, config.Y_RATIO_RANGE)
            
            augmentation_pipeline = A.Compose(
                [
                    GaussianRotate(mean=0, std=5, p=1.0),
                    A.RandomScale(scale_limit=scale_limit, p=1.0),
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
            )
            augmented_data = augmentation_pipeline(
                image=resized_embedded_img,
                mask=resized_mask, 
                bboxes=bounding_boxes,
                category_ids=labels
            )
            augmented_img = augmented_data['image']
            augmented_bboxes = augmented_data['bboxes']
            augmented_mask = augmented_data['mask']

            y_offset, x_offset = calculatePositionOffset(
                background_img.shape, augmented_img.shape, y_ratio, bbox_y_max, last_x_offset_set
            )
            y_end = min(y_offset + augmented_img.shape[0], background_img.shape[0])
            x_end = min(x_offset + augmented_img.shape[1], background_img.shape[1])

            offset_bboxes = apply_offsets_to_bboxes(augmented_bboxes, x_offset, y_offset)
            all_offset_bboxes.extend(offset_bboxes)
            all_labels.extend(labels)

            augmented_mask = cv2.cvtColor(augmented_mask, cv2.COLOR_BGR2GRAY)
            background_img[y_offset:y_end, x_offset:x_end] = cv2.bitwise_and(background_img[y_offset:y_end, x_offset:x_end], background_img[y_offset:y_end, x_offset:x_end], mask=cv2.bitwise_not(augmented_mask))
            augmented_img[:y_end-y_offset, :x_end-x_offset] = cv2.bitwise_and(augmented_img[:y_end-y_offset, :x_end-x_offset], augmented_img[:y_end-y_offset, :x_end-x_offset], mask=augmented_mask)  # 保留 mask 区域
            background_img[y_offset:y_end, x_offset:x_end] = cv2.add(background_img[y_offset:y_end, x_offset:x_end], augmented_img[:y_end-y_offset, :x_end-x_offset])

        output_img_name = 'aug_' + os.path.splitext(os.path.basename(background_img_path))[0] + '.jpg'
        output_annotation_name = 'aug_' + os.path.splitext(os.path.basename(background_img_path))[0] + '.xml'

        output_img_path = os.path.join(config.OUTPUT_IMG_DIR, output_img_name)
        output_annotation_path = os.path.join(config.OUTPUT_ANNOTATION_DIR, output_annotation_name)

        os.makedirs(config.OUTPUT_IMG_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_ANNOTATION_DIR, exist_ok=True)

        update_annotation_with_objects(
            background_annotation_path, output_annotation_path, all_offset_bboxes, all_labels
        )

        cv2.imwrite(output_img_path, background_img)

if __name__ == "__main__":
    main()
