import os
import tempfile

import cv2
import jsons
import numpy as np
import onnxruntime
from pdf2image import convert_from_path

import unstructured_inference
from unstructured_inference.inference.layout import DocumentLayout, LayoutElement, PageLayout
from unstructured_inference.visualize import vis
from unstructured_inference.yolox_functions import demo_postprocess, multiclass_nms
from unstructured_inference.yolox_functions import preproc as preprocess
from unstructured_inference.models import _get_model_loading_info


def yolox_local_inference(
    filename: str, type: str = "image", to_json: bool = False, keep_output: bool = False
):
    """This function creates a DocumentLayout from a file in local storage.
    type: accepted "image" and "pdf" files
    to_json: boolean indicating if transform DocumentLayout to json.
    keep_output: creates a folder with
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
                detections.append(image_processing(path, page_number=i, keep_output=keep_output))
            detectedDocument = DocumentLayout(detections)
            # Extract embedded text from PDF
            detectedDocument.parse_elements(filename, DPI=DPI)
    else:
        # Return a PageLayoutDocument
        detections = [image_processing(filename, keep_output=keep_output)]
        detectedDocument = DocumentLayout(detections)

    # NOTE (Benjamin): This version don't send images in json, could be
    # done in base64 in line 88, also, the decoding isn't full, as jsons.load
    # don't recreate PageLayout or LayoutElements
    if to_json:
        return jsons.dump(detectedDocument)

    return detectedDocument


def image_processing(page: str, page_number: int = 0, keep_output: bool = False) -> PageLayout:
    """
    Method runing YoloX for layout detection, returns a PageLayout
    page: path for image file with the image to process
    page_number: number asigned to the PageLayout returned
    keep_output: boolean indicating if result will be stored
    """
    # The model was trained and exported with this shape
    # TODO (benjamin): check other shapes for inference
    input_shape = (1024, 768)
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

        page_layout = PageLayout(
            number=page_number, image=None, layout=elements
        )  # TODO(benjamin): encode image as base64?
        if keep_output:
            if not os.path.exists(unstructured_inference.models.OUTPUT_DIR):
                os.makedirs(unstructured_inference.models.OUTPUT_DIR)
            # the  tmp_file laks of extension
            output_path = os.path.join(
                unstructured_inference.models.OUTPUT_DIR, os.path.basename(page_orig)
            )
            cv2.imwrite(output_path, annotated_image)

    return page_layout
