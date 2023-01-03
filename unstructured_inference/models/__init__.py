from typing import Tuple, Dict
from huggingface_hub import hf_hub_download

from unstructured_inference.models.detectron2 import load_model, Detectron2LayoutModel


def get_model(model: str) -> Detectron2LayoutModel:
    """Gets the model object by model name."""
    model_path, config_path, label_map = _get_model_loading_info(model)
    detector = load_model(config_path=config_path, model_path=model_path, label_map=label_map)

    return detector


def _get_model_loading_info(model: str) -> Tuple[str, str, Dict[int, str]]:
    """Gets local model binary and config locations and label map, downloading if necessary."""
    # TODO(alan): Find the right way to map model name to retrieval. It seems off that testing
    # needs to mock hf_hub_download.
    if model == "checkbox":
        repo_id = "unstructuredio/oer-checkbox"
        binary_fn = "detectron2_finetuned_oer_checkbox.pth"
        config_fn = "detectron2_oer_checkbox.json"
        model_path = hf_hub_download(repo_id, binary_fn)
        config_path = hf_hub_download(repo_id, config_fn)
        label_map = {0: "Unchecked", 1: "Checked"}
    else:
        raise UnknownModelException(f"Unknown model type: {model}")
    return model_path, config_path, label_map


class UnknownModelException(Exception):
    """Exception for the case where a model is called for with an unrecognized identifier."""

    pass
