import os
from typing import List, cast

import cv2
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from PIL import Image

from unstructured_inference.constants import Source
from unstructured_inference.inference.layoutelement import LayoutElement
from unstructured_inference.logger import logger
from unstructured_inference.models.unstructuredmodel import (
    UnstructuredObjectDetectionModel,
)


class UnstructuredSuperGradients(UnstructuredObjectDetectionModel):
    def predict(self, x: Image):
        """Predict using Super-Gradients model."""
        super().predict(x)
        return self.image_processing(x)

    def initialize(self, model_path: str, label_map: dict, input_shape: tuple):
        """Start inference session for SuperGradients model."""

        if not os.path.exists(model_path):
            logger.info("ONNX Model Path Does Not Exist!")
        self.model_path = model_path

        available_providers = C.get_available_providers()
        ordered_providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        providers = [provider for provider in ordered_providers if provider in available_providers]

        self.model = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.layout_classes = label_map

        self.input_shape = input_shape

    def image_processing(
        self,
        image: Image.Image,
    ) -> List[LayoutElement]:
        """Method runing SuperGradients Model for layout detection, returns a PageLayout"""
        # Not handling various input images right now
        # TODO (Pravin): check other shapes for inference
        input_shape = self.input_shape
        origin_img = np.array(image)
        img = preprocess(origin_img, input_shape)
        session = self.model
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        predictions = session.run(outputs, {inputs[0]: img})

        regions = []

        num_detections, pred_boxes, pred_scores, pred_classes = predictions
        for image_index in range(num_detections.shape[0]):
            for i in range(num_detections[image_index, 0]):
                class_id = pred_classes[image_index, i]
                prob = pred_scores[image_index, i]
                x1, y1, x2, y2 = pred_boxes[image_index, i]
                detected_class = self.layout_classes[str(class_id)]
                region = LayoutElement.from_coords(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    text=None,
                    type=detected_class,
                    prob=float(prob),
                    source=Source.SUPER_GRADIENTS,
                )
                regions.append(cast(LayoutElement, region))

        regions.sort(key=lambda element: element.bbox.y1)

        page_layout = regions

        return page_layout


def preprocess(origin_img, input_shape, swap=(0, 3, 1, 2)):
    """Preprocess image data before Super-Gradients Inputted Model
    Giving a generic preprocess function which simply resizes the image before prediction
    TODO(Pravin): Look into allowing user to specify their own pre-process function
    Which takes a numpy array image and returns a numpy array image
    """
    new_img = cv2.resize(origin_img, input_shape).astype(np.uint8)
    image_bchw = np.transpose(np.expand_dims(new_img, 0), swap)
    return image_bchw
