import os
from typing import List, Callable
import numpy as np
from PIL import Image
import yaml
import supervision as sv
from unstructured_inference.constants import Source
from unstructured_inference.inference.layoutelement import LayoutElement
from unstructured_inference.logger import logger
from unstructured_inference.models.unstructuredmodel import UnstructuredObjectDetectionModel
from super_gradients.training import models
import torch

class UnstructuredSuperGradients(UnstructuredObjectDetectionModel):
    def predict(self, x: Image):
        """Predict using Super-Gradients model."""
        super().predict(x)
        return self.image_processing(x)

    def initialize(self, model_arch: str, model_path: str, dataset_yaml_path: str, callback: Callable[[np.ndarray, models.sg_module.SgModule], sv.Detections]):
        """Start inference session for SuperGradients model."""
        if not os.path.exists(model_path):
            logger.info("Super Gradients Model Path Does Not Exist!")
        self.model_path = model_path
        
        with open(dataset_yaml_path, 'r') as file:
            dataset_yaml = yaml.safe_load(file)

        self.model = models.get(
        model_name=model_arch,
        num_classes=len(dataset_yaml['names']),
        checkpoint_path=model_path
        )
        
        label_map = {}
        for i,x in enumerate(dataset_yaml['names']):
            label_map[i] = x
        self.callback = callback
        self.layout_classes = label_map

    def image_processing(
        self,
        image: Image = None,
    ) -> List[LayoutElement]:
        """Method runing SuperGradients Model for layout detection, returns a PageLayout
        """
        # Not handling various input images right now
        # TODO (Pravin): check other shapes for inference
        origin_img = np.array(image)
        inference_model = self.model
        detections = self.callback(origin_img, inference_model)
        regions = []

        for det in detections:
            # Each detection should have (x1,y1,x2,y2,probability,class) format
            # being (x1,y1) the top left and (x2,y2) the bottom right
            x1, y1, x2, y2 = det[0].tolist()
            prob = det[2]
            class_id = det[3]
            detected_class = self.layout_classes[int(class_id)]
            region = LayoutElement(
                x1,
                y1,
                x2,
                y2,
                text=None,
                type=detected_class,
                prob=prob,
                source=Source.SUPER_GRADIENTS,
            )

            regions.append(region)

        regions.sort(key=lambda element: element.y1)

        page_layout = regions  # TODO(benjamin): encode image as base64?

        return page_layout
