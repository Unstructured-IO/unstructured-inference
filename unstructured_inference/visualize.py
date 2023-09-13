# Copyright (c) Megvii Inc. All rights reserved.
# Unstructured modified the original source code found at
# https://github.com/Megvii-BaseDetection/YOLOX/blob/ac379df3c97d1835ebd319afad0c031c36d03f36/yolox/utils/visualize.py
from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image
from PIL.ImageDraw import ImageDraw

from unstructured_inference.inference.elements import Rectangle


def draw_bbox(image: Image, rect: Rectangle, color: str = "red", width=1) -> Image:
    """Draws bounding box in image"""
    img = image.copy()
    draw = ImageDraw(img)
    topleft, _, bottomright, _ = rect.coordinates
    draw.rectangle((topleft, bottomright), outline=color, width=width)
    return img


# NOTE: in original files from YoloX 'draw_yolox_bounding_boxes' function is named "vis"
# TODO(alan): Need type hints here
def draw_yolox_bounding_boxes(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    """
    This function draws bounding boxes over the img argument, using
    boxes from detections from YoloX.
    img is a numpy array from cv2.imread()
    Scores refers to the probability of each detection.
    cls_ids are the class of each detection
    conf is the confidence required to draw the bounding box
    class_names is a list, where class_names[cls_ids[i]] should be the name
        for the i-th bounding box.
    """
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = f"{class_names[cls_id]}:{score * 100:.1f}%"
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def show_plot(
    image: Union[Image, np.ndarray],
    desired_width: Optional[int] = None,
):
    """
    Display an image using matplotlib with an optional desired width while maintaining the aspect
     ratio.

    Parameters:
    - image (Union[Image, np.ndarray]): An image in PIL Image format or a numpy ndarray format.
    - desired_width (Optional[int]): Desired width for the display size of the image.
        If provided, the height is calculated based on the original aspect ratio.
        If not provided, the image will be displayed with its original dimensions.

    Raises:
    - ValueError: If the provided image type is neither PIL Image nor numpy ndarray.

    Returns:
    - None: The function displays the image using matplotlib but does not return any value.
    """
    if isinstance(image, Image):
        image_width, image_height = image.size
    elif isinstance(image, np.ndarray):
        image_height, image_width, _ = image.shape
    else:
        raise ValueError("Unsupported Image Type")

    if desired_width:
        # Calculate the desired height based on the original aspect ratio
        aspect_ratio = image_width / image_height
        desired_height = desired_width / aspect_ratio

        # Create a figure with the desired size and aspect ratio
        fig, ax = plt.subplots(figsize=(desired_width, desired_height))
    else:
        # Create figure and axes
        fig, ax = plt.subplots()
    # Display the image
    ax.imshow(image)
    plt.show()


_COLORS = np.array(
    [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
        [0.635, 0.078, 0.184],
        [0.300, 0.300, 0.300],
        [0.600, 0.600, 0.600],
        [1.000, 0.000, 0.000],
        [1.000, 0.500, 0.000],
        [0.749, 0.749, 0.000],
        [0.000, 1.000, 0.000],
        [0.000, 0.000, 1.000],
        [0.667, 0.000, 1.000],
        [0.333, 0.333, 0.000],
        [0.333, 0.667, 0.000],
        [0.333, 1.000, 0.000],
        [0.667, 0.333, 0.000],
        [0.667, 0.667, 0.000],
        [0.667, 1.000, 0.000],
        [1.000, 0.333, 0.000],
        [1.000, 0.667, 0.000],
        [1.000, 1.000, 0.000],
        [0.000, 0.333, 0.500],
        [0.000, 0.667, 0.500],
        [0.000, 1.000, 0.500],
        [0.333, 0.000, 0.500],
        [0.333, 0.333, 0.500],
        [0.333, 0.667, 0.500],
        [0.333, 1.000, 0.500],
        [0.667, 0.000, 0.500],
        [0.667, 0.333, 0.500],
        [0.667, 0.667, 0.500],
        [0.667, 1.000, 0.500],
        [1.000, 0.000, 0.500],
        [1.000, 0.333, 0.500],
        [1.000, 0.667, 0.500],
        [1.000, 1.000, 0.500],
        [0.000, 0.333, 1.000],
        [0.000, 0.667, 1.000],
        [0.000, 1.000, 1.000],
        [0.333, 0.000, 1.000],
        [0.333, 0.333, 1.000],
        [0.333, 0.667, 1.000],
        [0.333, 1.000, 1.000],
        [0.667, 0.000, 1.000],
        [0.667, 0.333, 1.000],
        [0.667, 0.667, 1.000],
        [0.667, 1.000, 1.000],
        [1.000, 0.000, 1.000],
        [1.000, 0.333, 1.000],
        [1.000, 0.667, 1.000],
        [0.333, 0.000, 0.000],
        [0.500, 0.000, 0.000],
        [0.667, 0.000, 0.000],
        [0.833, 0.000, 0.000],
        [1.000, 0.000, 0.000],
        [0.000, 0.167, 0.000],
        [0.000, 0.333, 0.000],
        [0.000, 0.500, 0.000],
        [0.000, 0.667, 0.000],
        [0.000, 0.833, 0.000],
        [0.000, 1.000, 0.000],
        [0.000, 0.000, 0.167],
        [0.000, 0.000, 0.333],
        [0.000, 0.000, 0.500],
        [0.000, 0.000, 0.667],
        [0.000, 0.000, 0.833],
        [0.000, 0.000, 1.000],
        [0.000, 0.000, 0.000],
        [0.143, 0.143, 0.143],
        [0.286, 0.286, 0.286],
        [0.429, 0.429, 0.429],
        [0.571, 0.571, 0.571],
        [0.714, 0.714, 0.714],
        [0.857, 0.857, 0.857],
        [0.000, 0.447, 0.741],
        [0.314, 0.717, 0.741],
        [0.50, 0.5, 0],
    ],
).astype(np.float32)
