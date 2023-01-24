from typing import Tuple, Dict, Optional, List, Any
from huggingface_hub import hf_hub_download

from unstructured_inference.models.detectron2 import (
    load_model,
    Detectron2LayoutModel,
    DEFAULT_LABEL_MAP,
    DEFAULT_EXTRA_CONFIG,
)


def get_model(model: Optional[str] = None) -> Detectron2LayoutModel:
    """Gets the model object by model name."""
    model_path, config_path, label_map, extra_config = _get_model_loading_info(model)
    detector = load_model(
        config_path=config_path,
        model_path=model_path,
        label_map=label_map,
        extra_config=extra_config,
    )
    return detector


def _get_model_loading_info(
    model: Optional[str],
) -> Tuple[str, str, Dict[int, str], Optional[List[Any]]]:
    """Gets local model binary and config locations and label map, downloading if necessary."""
    # TODO(alan): Find the right way to map model name to retrieval. It seems off that testing
    # needs to mock hf_hub_download.
    if model is None:
        repo_id = "layoutparser/detectron2"
        binary_fn = "PubLayNet/faster_rcnn_R_50_FPN_3x/model_final.pth"
        config_fn = "PubLayNet/faster_rcnn_R_50_FPN_3x/config.yml"
        model_path = hf_hub_download(repo_id, binary_fn)
        config_path = hf_hub_download(repo_id, config_fn)
        label_map = DEFAULT_LABEL_MAP
        extra_config = DEFAULT_EXTRA_CONFIG
    elif model == "checkbox":
        repo_id = "unstructuredio/oer-checkbox"
        binary_fn = "detectron2_finetuned_oer_checkbox.pth"
        config_fn = "detectron2_oer_checkbox.json"
        model_path = hf_hub_download(repo_id, binary_fn)
        config_path = hf_hub_download(repo_id, config_fn)
        label_map = {0: "Unchecked", 1: "Checked"}
        extra_config = None
    else:
        raise UnknownModelException(f"Unknown model type: {model}")
    return model_path, config_path, label_map, extra_config


class UnknownModelException(Exception):
    """Exception for the case where a model is called for with an unrecognized identifier."""

    pass
