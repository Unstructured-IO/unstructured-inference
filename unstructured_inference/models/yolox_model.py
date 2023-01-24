import os
import tempfile

import cv2
import jsons
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from pdf2image import convert_from_path

from unstructured_inference.inference.layout import DocumentLayout, LayoutElement, PageLayout
from unstructured_inference.visualize import vis
from unstructured_inference.yolox_functions import demo_postprocess, multiclass_nms
from unstructured_inference.yolox_functions import preproc as preprocess

# NOTE(benjamin) Repository and file to download from hugging_face
REPO_ID = "unstructuredio/yolo_x_layout"
FILENAME = "yolox_l0.05.onnx"

LAYOUT_CLASSES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

output_dir = "outputs/"


def yolox_local_inference(filename, type="image", to_json=False, keep_output=False):
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
                detections.append(image_processing(path, page_number=i, keep_output=keep_output))
            detectedDocument = DocumentLayout(detections)
            # Extract embedded text from PDF
            detectedDocument.parse_elements(filename, DPI=DPI)
    else:
        # Return a PageLayoutDocument
        detections = image_processing(filename, keep_output=keep_output)
        detectedDocument = DocumentLayout([detections])

    # NOTE (Benjamin): This version don't send images in json, could be
    # done in base64 in line 88, also, the decoding isn't full, as jsons.load
    # don't recreate PageLayout or LayoutElements
    if to_json:
        return jsons.dump(detectedDocument)

    return detectedDocument


def image_processing(page, page_number=0, keep_output=False) -> PageLayout:
    """
    Method runing YoloX for layout detection, returns a PageLayout
    """
    YOLOX_MODEL = hf_hub_download(REPO_ID, FILENAME)
    # The model was trained and exported with this shape
    # TODO (benjamin): check other shapes for inference
    input_shape = (1024, 768)
    origin_img = cv2.imread(page)
    img, ratio = preprocess(origin_img, input_shape)
    page_orig = page
    session = onnxruntime.InferenceSession(YOLOX_MODEL)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape, p6=False)[0]  # TODO(benjamin): check for p6

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        annotated_image = vis(
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

    page = PageLayout(
        number=page_number, image=None, layout=elements
    )  # TODO(benjamin): encode image as base64?

    if keep_output:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # the  tmp_file laks of extension
        output_path = os.path.join(output_dir, os.path.basename(page_orig))
        cv2.imwrite(output_path, annotated_image)

    return page
