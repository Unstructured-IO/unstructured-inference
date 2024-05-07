from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Union

from layoutparser.models.detectron2.layoutmodel import (
    Detectron2LayoutModel,
    is_detectron2_available,
)
from layoutparser.models.model_config import LayoutModelConfig
from PIL import Image as PILImage

from unstructured_inference.constants import ElementType
from unstructured_inference.inference.layoutelement import LayoutElement
from unstructured_inference.logger import logger
from unstructured_inference.models.unstructuredmodel import (
    UnstructuredObjectDetectionModel,
)
from unstructured_inference.utils import (
    LazyDict,
    LazyEvaluateInfo,
    download_if_needed_and_get_local_path,
)

DETECTRON_CONFIG: Final = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
DEFAULT_LABEL_MAP: Final[Dict[int, str]] = {
    0: ElementType.TEXT,
    1: ElementType.TITLE,
    2: ElementType.LIST,
    3: ElementType.TABLE,
    4: ElementType.FIGURE,
}
DEFAULT_EXTRA_CONFIG: Final[List[Any]] = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]


# NOTE(alan): Entries are implemented as LazyDicts so that models aren't downloaded until they are
# needed.
MODEL_TYPES = {
    "detectron2_lp": LazyDict(
        model_path=LazyEvaluateInfo(
            download_if_needed_and_get_local_path,
            "layoutparser/detectron2",
            "PubLayNet/faster_rcnn_R_50_FPN_3x/model_final.pth",
        ),
        config_path=LazyEvaluateInfo(
            download_if_needed_and_get_local_path,
            "layoutparser/detectron2",
            "PubLayNet/faster_rcnn_R_50_FPN_3x/config.yml",
        ),
        label_map=DEFAULT_LABEL_MAP,
        extra_config=DEFAULT_EXTRA_CONFIG,
    ),
    "checkbox": LazyDict(
        model_path=LazyEvaluateInfo(
            download_if_needed_and_get_local_path,
            "unstructuredio/oer-checkbox",
            "detectron2_finetuned_oer_checkbox.pth",
        ),
        config_path=LazyEvaluateInfo(
            download_if_needed_and_get_local_path,
            "unstructuredio/oer-checkbox",
            "detectron2_oer_checkbox.json",
        ),
        label_map={0: "Unchecked", 1: "Checked"},
        extra_config=None,
    ),
}


class UnstructuredDetectronModel(UnstructuredObjectDetectionModel):
    """Unstructured model wrapper for Detectron2LayoutModel."""

    def predict(self, x: PILImage.Image):
        """Makes a prediction using detectron2 model."""
        super().predict(x)
        prediction = self.model.detect(x)
        return [LayoutElement.from_lp_textblock(block) for block in prediction]

    def initialize(
        self,
        config_path: Union[str, Path, LayoutModelConfig],
        model_path: Optional[Union[str, Path]] = None,
        label_map: Optional[Dict[int, str]] = None,
        extra_config: Optional[list] = None,
        device: Optional[str] = None,
    ):
        """Loads the detectron2 model using the specified parameters"""

        if not is_detectron2_available():
            raise ImportError(
                "Failed to load the Detectron2 model. Ensure that the Detectron2 "
                "module is correctly installed.",
            )

        config_path_str = str(config_path)
        model_path_str: Optional[str] = None if model_path is None else str(model_path)
        logger.info("Loading the Detectron2 layout model ...")
        self.model = Detectron2LayoutModel(
            config_path_str,
            model_path=model_path_str,
            label_map=label_map,
            extra_config=extra_config,
            device=device,
        )
