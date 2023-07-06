import os
import time
from typing import List, cast

import cv2
import numpy as np
import pytesseract

from unstructured_inference.inference import layout
from unstructured_inference.inference.elements import TextRegion
from unstructured_inference.models.base import get_model
from unstructured_inference.models.unstructuredmodel import UnstructuredObjectDetectionModel, \
    UnstructuredElementExtractionModel


def run_ocr_with_layout_detection(
    images,
    mode="individual_blocks",
    printable=True,
    drawable=True,
    output_dir="",
):
    model = get_model()
    if isinstance(model, UnstructuredObjectDetectionModel):
        detection_model = model
        element_extraction_model = None
        print("model_type: UnstructuredObjectDetectionModel")
    elif isinstance(model, UnstructuredElementExtractionModel):
        detection_model = None
        element_extraction_model = model
        print("model_type: UnstructuredElementExtractionModel")
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    total_text_extraction_infer_time = 0
    total_text = ""
    for i, image in enumerate(images):
        page = layout.PageLayout(
            number=i+1,
            image=image,
            layout=None,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model
        )

        inferred_layout: List[TextRegion] = cast(List[TextRegion], page.detection_model(page.image))

        cv_img = np.array(image)
        if drawable:
            for el in inferred_layout:
                pt1 = [int(el.x1), int(el.y1)]
                pt2 = [int(el.x2), int(el.y2)]
                cv2.rectangle(
                    img=cv_img,
                    pt1=pt1, pt2=pt2,
                    color=(0, 0, 255),
                    thickness=4,
                    lineType=None,
                )

        if mode == "individual_blocks":
            # OCR'ing individual blocks (current approach)
            text_extraction_start_time = time.time()

            elements = page.get_elements_from_layout(inferred_layout)

            text_extraction_infer_time = time.time() - text_extraction_start_time

            if printable:
                print(f"page: {i+1} - n_layout_elements: {len(inferred_layout)} - "
                      f"text_extraction_infer_time: {text_extraction_infer_time}")

            total_text_extraction_infer_time += text_extraction_infer_time

            page_text = ""
            for el in elements:
                page_text += el.text
            total_text += page_text

        elif mode == "entire_page":
            # OCR'ing entire page (new approach to implement)
            pass
        else:
            print("Invalid mode")

        if drawable:
            f_path = os.path.join(output_dir, f"ocr_individual_blocks_page_{i + 1}.jpg")
            cv2.imwrite(f_path, cv_img)

    return total_text_extraction_infer_time, total_text


def run_ocr(
    images,
    printable=True
):
    total_text_extraction_infer_time = 0
    total_text = ""
    for i, image in enumerate(images):
        text_extraction_start_time = time.time()

        page_text = pytesseract.image_to_string(image)

        text_extraction_infer_time = time.time() - text_extraction_start_time

        if printable:
            print(f"page: {i + 1} - text_extraction_infer_time: {text_extraction_infer_time}")

        total_text_extraction_infer_time += text_extraction_infer_time
        total_text += page_text

    return total_text_extraction_infer_time, total_text
