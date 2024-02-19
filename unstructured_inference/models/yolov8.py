from typing import List, cast

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.ops import nms

from unstructured_inference.constants import ElementType, Source
from unstructured_inference.inference.layoutelement import LayoutElement
from unstructured_inference.models.unstructuredmodel import UnstructuredObjectDetectionModel
from unstructured_inference.utils import LazyDict, LazyEvaluateInfo
from ultralytics import YOLO

YOLOv8_LABEL_MAP = {
    0: ElementType.CAPTION,
    1: ElementType.FOOTNOTE,
    2: ElementType.FORMULA,
    3: ElementType.LIST_ITEM,
    4: ElementType.PAGE_FOOTER,
    5: ElementType.PAGE_HEADER,
    6: ElementType.PICTURE,
    7: ElementType.SECTION_HEADER,
    8: ElementType.TABLE,
    9: ElementType.TEXT,
    10: ElementType.TITLE,
}
label_to_color = {
    ElementType.CAPTION: "black",
    ElementType.FOOTNOTE: "cyan",
    ElementType.FORMULA: "black",
    ElementType.LIST_ITEM: "green",
    ElementType.PAGE_FOOTER: "blue",
    ElementType.PAGE_HEADER: "yellow",
    ElementType.PICTURE: "black",
    ElementType.SECTION_HEADER: "purple",
    ElementType.TABLE: "black",
    ElementType.TEXT: "black",
    ElementType.TITLE: "red",
}

model = YOLO('/home/joao/yolov8n/weights/best.pt')
MODEL_TYPES = {
    "yolov8n": LazyDict(
        model_path=LazyEvaluateInfo(
            hf_hub_download,
            "neuralshift/doc-layout-yolov8n",
            "weights/best.pt",
        ),
        label_map=YOLOv8_LABEL_MAP,
    ),
    "yolov8s": LazyDict(
        model_path=LazyEvaluateInfo(
            hf_hub_download,
            "neuralshift/doc-layout-yolov8s",
            "weights/best.pt",
        ),
        label_map=YOLOv8_LABEL_MAP,
    ),
}


class UnstructuredYolov8Model(UnstructuredObjectDetectionModel):
    def predict(self, x: Image):
        """Predict using YoloX model."""
        super().predict(x)
        return self.image_processing(x)

    def initialize(self, model_path: str, label_map: dict):
        """Start inference session for YoloX model."""
        self.model = YOLO(model=model_path)
        self.layout_classes = label_map

    def image_processing(
        self,
        image: Image = None,
    ) -> List[LayoutElement]:
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
        input_shape = (640, 640)
        processed_image = image.resize(input_shape, Image.BILINEAR)
        ratio = np.array(input_shape) / np.array(image.size)

        # NMS
        boxes = self.model(processed_image, verbose=False)[0].boxes
        valid_boxes = nms(boxes.xyxy, boxes.conf, 0.1)
        boxes = boxes[valid_boxes]
        boxes = boxes[boxes.conf > 0.3]

        regions = sorted([
            LayoutElement.from_coords(
                box.xyxy[0][0] / ratio[0],
                box.xyxy[0][1] / ratio[1],
                box.xyxy[0][2] / ratio[0],
                box.xyxy[0][3] / ratio[1],
                text=None,
                type=self.layout_classes[int(box.cls.item())],
                prob=box.conf.item(),
                source=Source.YOLOv8,
            ) for box in boxes
        ], key=lambda element: element.bbox.y1)

        page_layout = cast(List[LayoutElement], regions)  # TODO(benjamin): encode image as base64?
        
        return page_layout
