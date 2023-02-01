from typing import Tuple, Dict
from huggingface_hub import hf_hub_download


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
    elif model == "yolox":
        # NOTE(benjamin) Repository and file to download from hugging_face
        repo_id = "unstructuredio/yolo_x_layout"
        binary_fn = "yolox_l0.05.onnx"
        model_path = hf_hub_download(repo_id, binary_fn)
        label_map = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title",
        }
        # NOTE(benjamin): just for acomplish with previous version of this function
        config_path = None

    else:
        raise UnknownModelException(f"Unknown model type: {model}")
    # NOTE(benjamin): Maybe could return a dictionary intead this set of variables
    return model_path, config_path, label_map


class UnknownModelException(Exception):
    """Exception for the case where a model is called for with an unrecognized identifier."""

    pass
