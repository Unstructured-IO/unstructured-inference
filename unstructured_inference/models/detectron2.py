from typing import Final, Optional, Union, List, Dict, Any
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


DETECTRON_CONFIG: Final = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
DEFAULT_LABEL_MAP: Final[Dict[int, str]] = {
    0: "Text",
    1: "Title",
    2: "List",
    3: "Table",
    4: "Figure",
}
DEFAULT_EXTRA_CONFIG: Final[List[Any]] = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]


# NOTE(alan): Entries are implemented as LazyDicts so that models aren't downloaded until they are
# needed.
MODEL_TYPES = {
    None: LazyDict(
        model_path=LazyEvaluateInfo(
            hf_hub_download,
            "unstructuredio/detectron2_faster_rcnn_R_50_FPN_3x",
            "model.onnx",
        ),
        label_map=DEFAULT_LABEL_MAP,
        # extra_config=DEFAULT_EXTRA_CONFIG,
    ),
    "checkbox": LazyDict(
        model_path=LazyEvaluateInfo(
            hf_hub_download, "unstructuredio/oer-checkbox", "detectron2_finetuned_oer_checkbox.pth"
        ),
        config_path=LazyEvaluateInfo(
            hf_hub_download, "unstructuredio/oer-checkbox", "detectron2_oer_checkbox.json"
        ),
        label_map={0: "Unchecked", 1: "Checked"},
        extra_config=None,
    ),
}


class UnstructuredDetectronModel(UnstructuredModel):
    """Unstructured model wrapper for Detectron2LayoutModel."""

    def predict(self, x: Image):
        """Makes a prediction using detectron2 model."""
        super().predict(x)
        prediction = self.image_processing(x)
        return prediction

    def initialize(
        self,
        model_path: Optional[Union[str, Path]] = None,
        label_map: Optional[Dict[int, str]] = None,
    ):
        """Loads the detectron2 model using the specified parameters"""
        logger.info("Loading the Detectron2 layout model ...")
        self.model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.label_map = label_map

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
        bboxes, labels, confidence, _ = output
        bboxes = bboxes.tolist()
        labels = labels.tolist()
        confidence = confidence.tolist()

        regions = []
        for (x1, y1, w, h), label, conf in zip(bboxes, labels, confidence):
            # Each detection should have (x1,y1,x2,y2,probability,class) format
            # being (x1,y1) the top left and (x2,y2) the bottom right
            detected_class = self.label_map[int(label)]
            print(detected_class, conf)
            if conf >= 0.8:
                region = LayoutElement(x1, y1, x1 + w, y1 + h, text=None, type=detected_class)

                regions.append(region)

        regions.sort(key=lambda element: element.y1)

        return regions
