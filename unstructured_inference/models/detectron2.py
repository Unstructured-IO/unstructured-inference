from typing import Final, Optional, Union, List, Dict, Any
from pathlib import Path

from layoutparser.models.detectron2.layoutmodel import (
    is_detectron2_available,
    Detectron2LayoutModel,
)
from layoutparser.models.model_config import LayoutModelConfig

from unstructured_inference.logger import logger


DETECTRON_CONFIG: Final = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
DEFAULT_LABEL_MAP: Final[Dict[int, str]] = {
    0: "Text",
    1: "Title",
    2: "List",
    3: "Table",
    4: "Figure",
}
DEFAULT_EXTRA_CONFIG: Final[List[Any]] = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]


def load_model(
    config_path: Union[str, Path, LayoutModelConfig],
    model_path: Optional[Union[str, Path]] = None,
    label_map: Optional[Dict[int, str]] = None,
    extra_config: Optional[list] = None,
    device: Optional[str] = None,
) -> Detectron2LayoutModel:
    """Loads the detectron2 model using the specified parameters"""

    if not is_detectron2_available():
        raise ImportError(
            "Failed to load the Detectron2 model. Ensure that the Detectron2 "
            "module is correctly installed."
        )

    config_path_str = str(config_path)
    model_path_str: Optional[str] = None if model_path is None else str(model_path)
    logger.info("Loading the Detectron2 layout model ...")
    model = Detectron2LayoutModel(
        config_path_str,
        model_path=model_path_str,
        label_map=label_map,
        extra_config=extra_config,
        device=device,
    )
    return model


def load_default_model() -> Detectron2LayoutModel:
    """Loads the detectron2 model using default parameters"""
    return load_model(
        config_path=DETECTRON_CONFIG, label_map=DEFAULT_LABEL_MAP, extra_config=DEFAULT_EXTRA_CONFIG
    )
