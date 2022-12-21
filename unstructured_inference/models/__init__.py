import os

from unstructured_inference import MODEL_LOCATION
from unstructured_inference.models.detectron2 import load_model, Detectron2LayoutModel


def get_model(model: str) -> Detectron2LayoutModel:
    if model == "checkbox":
        model_path = os.path.join(MODEL_LOCATION, "detectron2_finetuned_oer_checkbox.pth")
        config_path = os.path.join(MODEL_LOCATION, "detectron2_oer_checkbox.json")
        label_map = {0: "Unchecked", 1: "Checked"}
        detector = load_model(config_path=config_path, model_path=model_path, label_map=label_map)
    else:
        raise ValueError(f"Unknown model type: {model}")

    return detector
