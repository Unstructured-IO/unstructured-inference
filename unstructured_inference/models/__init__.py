import json
from typing import Tuple, Dict
from huggingface_hub import hf_hub_download


def _get_model_loading_info(model: str) -> Tuple[str, str, Dict[int, str]]:
    """Gets local model binary and config locations and label map, downloading if necessary."""
    # TODO(alan): Find the right way to map model name to retrieval. It seems off that testing
    # needs to mock hf_hub_download.
    # NOTE(benjamin) Repository and file to download from hugging_face
    hf_names = {
        "yolox": ("yolox_l0.05.onnx", "label_map.json"),
        "yolox_tiny": ("yolox_tiny.onnx", "label_map.json"),
    }
    try:
        repo_id = "unstructuredio/yolo_x_layout"
        binary_fn, label_path = hf_names[model]
        model_path = hf_hub_download(repo_id, binary_fn)
        # As JSON only encode keys as strings, we need to parse strings to ints
        label_map = json.load(
            open(hf_hub_download(repo_id, label_path), "r"),
            object_hook=lambda d: {int(k): v for k, v in d.items()},
        )
        # NOTE(benjamin): just for acomplish with previous version of this function
        config_path = ""
    except KeyError:
        raise UnknownModelException(f"Unknown model type: {model}")
    # NOTE(benjamin): Maybe could return a dictionary intead this set of variables
    return model_path, config_path, label_map


class UnknownModelException(Exception):
    """Exception for the case where a model is called for with an unrecognized identifier."""

    pass
