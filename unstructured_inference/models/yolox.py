# Copyright (c) Megvii, Inc. and its affiliates.
# Unstructured modified the original source code found at:
# https://github.com/Megvii-BaseDetection/YOLOX/blob/237e943ac64aa32eb32f875faa93ebb18512d41d/yolox/data/data_augment.py
# https://github.com/Megvii-BaseDetection/YOLOX/blob/ac379df3c97d1835ebd319afad0c031c36d03f36/yolox/utils/demo_utils.py
import os
import tempfile
from typing import Optional
from PIL import Image

import cv2
import numpy as np
import onnxruntime
from pdf2image import convert_from_path

from unstructured_inference.inference.layout import DocumentLayout, LayoutElement, PageLayout
from unstructured_inference.models import _get_model_loading_info
from unstructured_inference.visualize import draw_bounding_boxes


def yolox_local_inference(
    filename: str,
    type: str = "image",
    use_ocr=False,
    output_directory: Optional[str] = None,
) -> DocumentLayout:
    """This function creates a DocumentLayout from a file in local storage.
    Parameters
    ----------
    type
        Accepted "image" and "pdf" files
    use_ocr:
        For pdf without embedded text, this function will use OCR for
        text extraction
    output_directory
        Default 'None', if specified, the output of YoloX model will be
        drawed over page images at this folder
    """
    DPI = 500
    pages_paths = []
    detections = []
    detectedDocument = None
    if type == "pdf":
        with tempfile.TemporaryDirectory() as tmp_folder:
            pages_paths = convert_from_path(
                filename, dpi=DPI, output_folder=tmp_folder, paths_only=True
            )
            for i, path in enumerate(pages_paths):
                # Return a dict of {n-->PageLayoutDocument}
                detections.append(
                    image_processing(path, page_number=i, output_directory=output_directory)
                )
            detectedDocument = DocumentLayout(detections)
            if use_ocr:
                for n, page_path in enumerate(pages_paths):
                    detectedDocument.parse_image_elements(page_path, n)
            else:
                # Extract embedded text from PDF
                detectedDocument.parse_elements(filename, DPI=DPI)
    else:
        # Return a PageLayoutDocument
        detections = [
            image_processing(
                filename, origin_img=None, page_number=0, output_directory=output_directory
            )
        ]
        detectedDocument = DocumentLayout(detections)
        detectedDocument.parse_image_elements(filename, 0)

    return detectedDocument


def image_processing(
    page: str,
    origin_img: Image = None,
    page_number: int = 0,
    output_directory: Optional[str] = None,
) -> PageLayout:
    """Method runing YoloX for layout detection, returns a PageLayout
    parameters
    ----------
    page
        Path for image file with the image to process
    origin_img
        If specified, an Image object for process with YoloX model
    page_number
        Number asigned to the PageLayout returned
    output_directory
        Boolean indicating if result will be stored
    """
    # The model was trained and exported with this shape
    # TODO (benjamin): check other shapes for inference
    input_shape = (1024, 768)
    if origin_img and page:
        raise ValueError("Just one of the arguments allowed 'page' or 'origin_img'")
    if not origin_img:
        origin_img = cv2.imread(page)
    img, ratio = preprocess(origin_img, input_shape)
    page_orig = page
    # TODO (benjamin): We should use models.get_model() but currenly returns Detectron model
    model_path, _, LAYOUT_CLASSES = _get_model_loading_info("yolox")
    session = onnxruntime.InferenceSession(model_path)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape, p6=False)[
        0
    ]  # TODO(benjamin): check for p6

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

    # If dets is None, the page is created empty, else this object will be replaced
    page_layout = PageLayout(number=page_number, image=None, layout=[])

    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        annotated_image = draw_bounding_boxes(
            origin_img,
            final_boxes,
            final_scores,
            final_cls_inds,
            conf=0.3,
            class_names=LAYOUT_CLASSES,
        )

        elements = []
        # Each detection should have (x1,y1,x2,y2,probability,class) format
        # being (x1,y1) the top left and (x2,y2) the bottom right

        for det in dets:
            detection = det.tolist()
            detection[-1] = LAYOUT_CLASSES[int(detection[-1])]
            element = LayoutElement(
                type=detection[-1],
                coordinates=[(detection[0], detection[1]), (detection[2], detection[3])],
                text=" ",
            )

            elements.append(element)

        elements.sort(key=lambda element: element.coordinates[0][1])

        page_layout = PageLayout(
            number=page_number, image=None, layout=elements
        )  # TODO(benjamin): encode image as base64?
        if output_directory:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            # the  tmp_file laks of extension
            output_path = os.path.join(output_directory, os.path.basename(page_orig))
            cv2.imwrite(output_path, annotated_image)

    return page_layout


# Note: preprocess function was named preproc on original source


def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    # TODO(benjamin): check for non-class agnostic
    # if class_agnostic:
    nms_method = multiclass_nms_class_agnostic
    # else:
    #    nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep
