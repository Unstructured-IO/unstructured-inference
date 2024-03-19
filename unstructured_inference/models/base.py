import json
import os
from typing import Dict, Optional, Type

from unstructured_inference.models.chipper import MODEL_TYPES as CHIPPER_MODEL_TYPES
from unstructured_inference.models.chipper import UnstructuredChipperModel
from unstructured_inference.models.detectron2 import (
    MODEL_TYPES as DETECTRON2_MODEL_TYPES,
)
from unstructured_inference.models.detectron2 import UnstructuredDetectronModel
from unstructured_inference.models.detectron2onnx import (
    MODEL_TYPES as DETECTRON2_ONNX_MODEL_TYPES,
)
from unstructured_inference.models.detectron2onnx import UnstructuredDetectronONNXModel
from unstructured_inference.models.unstructuredmodel import UnstructuredModel
from unstructured_inference.models.yolox import MODEL_TYPES as YOLOX_MODEL_TYPES
from unstructured_inference.models.yolox import UnstructuredYoloXModel

DEFAULT_MODEL = "yolox"

models: Dict[str, UnstructuredModel] = {}

model_class_map: Dict[str, Type[UnstructuredModel]] = {
    **{name: UnstructuredDetectronModel for name in DETECTRON2_MODEL_TYPES},
    **{name: UnstructuredDetectronONNXModel for name in DETECTRON2_ONNX_MODEL_TYPES},
    **{name: UnstructuredYoloXModel for name in YOLOX_MODEL_TYPES},
    **{name: UnstructuredChipperModel for name in CHIPPER_MODEL_TYPES},
}


def get_model(model_name: Optional[str] = None) -> UnstructuredModel:
    """Gets the model object by model name."""
    # TODO(alan): These cases are similar enough that we can probably do them all together with
    # importlib

    global models

    if model_name is None:
        default_name_from_env = os.environ.get("UNSTRUCTURED_DEFAULT_MODEL_NAME")
        model_name = default_name_from_env if default_name_from_env is not None else DEFAULT_MODEL

    if model_name in models:
        return models[model_name]

    initialize_param_json = os.environ.get("UNSTRUCTURED_DEFAULT_MODEL_INITIALIZE_PARAMS_JSON_PATH")
    if initialize_param_json is not None:
        with open(initialize_param_json) as fp:
            initialize_params = json.load(fp)
            label_map_int_keys = {
                int(key): value for key, value in initialize_params["label_map"].items()
            }
            initialize_params["label_map"] = label_map_int_keys
    else:
        if model_name in DETECTRON2_MODEL_TYPES:
            initialize_params = DETECTRON2_MODEL_TYPES[model_name]
        elif model_name in DETECTRON2_ONNX_MODEL_TYPES:
            initialize_params = DETECTRON2_ONNX_MODEL_TYPES[model_name]
        elif model_name in YOLOX_MODEL_TYPES:
            initialize_params = YOLOX_MODEL_TYPES[model_name]
        elif model_name in CHIPPER_MODEL_TYPES:
            initialize_params = CHIPPER_MODEL_TYPES[model_name]
        else:
            raise UnknownModelException(f"Unknown model type: {model_name}")

    model: UnstructuredModel = model_class_map[model_name]()

    model.initialize(**initialize_params)
    models[model_name] = model
    return model


class UnknownModelException(Exception):
    """Exception for the case where a model is called for with an unrecognized identifier."""

    pass
