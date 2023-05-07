from typing import Final, Optional, Union, List, Dict
from pathlib import Path

from PIL import Image
from huggingface_hub import hf_hub_download

from unstructured_inference.logger import logger
from unstructured_inference.inference.layoutelement import LayoutElement
from unstructured_inference.models.unstructuredmodel import UnstructuredModel
from unstructured_inference.utils import LazyDict, LazyEvaluateInfo
import onnxruntime
import numpy as np
import cv2


DEFAULT_LABEL_MAP: Final[Dict[int, str]] = {
    0: "Text",
    1: "Title",
    2: "List",
    3: "Table",
    4: "Figure",
}


# NOTE(alan): Entries are implemented as LazyDicts so that models aren't downloaded until they are
# needed.
MODEL_TYPES: Dict[Optional[str], LazyDict] = {
    None: LazyDict(
        model_path=LazyEvaluateInfo(
            hf_hub_download,
            "unstructuredio/detectron2_faster_rcnn_R_50_FPN_3x",
            "model.onnx",
        ),
        label_map=DEFAULT_LABEL_MAP,
        confidence_threshold=0.8,
    ),
}


class UnstructuredDetectronModel(UnstructuredModel):
    """Unstructured model wrapper for Detectron2LayoutModel."""

    def predict(self, x: Image.Image):
        """Makes a prediction using detectron2 model."""
        super().predict(x)
        prediction = self.image_processing(x)
        return prediction

    def initialize(
        self,
        model_path: Union[str, Path],
        label_map: Dict[int, str],
        confidence_threshold: Optional[float] = None,
    ):
        """Loads the detectron2 model using the specified parameters"""
        logger.info("Loading the Detectron2 layout model ...")
        self.model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.label_map = label_map
        if confidence_threshold is None:
            confidence_threshold = 0.5
        self.confidence_threshold = confidence_threshold

    def image_processing(
        self,
        image: Image.Image,
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
        # The model was trained and exported with this shape
        # TODO (benjamin): check other shapes for inference
        input_shape = (800, 1035)  # detectron2 specific
        img = np.array(image)
        # TODO (benjamin): We should use models.get_model() but currenly returns Detectron model
        session = self.model
        # onnx input expected
        # [3,1035,800]
        img = cv2.resize(
            img,
            input_shape,
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        img = img.transpose(2, 0, 1)
        ort_inputs = {session.get_inputs()[0].name: img}
        output = session.run(None, ort_inputs)
        # output[0] seems like bounding boxes (in pixels, not sure if original size
        # or in (1035,800))
        # output boxes seems like xywh due similar 3rd entry in all elements (same width)
        # output[1] seems like labels for bboxes
        # output[2] seems like confidence score for each label
        # output[3] seems like image size (it's fixed to (1035,800) so this info is useless now)
        bboxes, labels, confidence_scores, _ = output

        regions = []
        for (x1, y1, x2, y2), label, conf in zip(bboxes, labels, confidence_scores):
            detected_class = self.label_map[int(label)]
            if conf >= self.confidence_threshold:
                region = LayoutElement(x1, y1, x2, y2, text=None, type=detected_class)

                regions.append(region)

        regions.sort(key=lambda element: element.y1)

        return regions
