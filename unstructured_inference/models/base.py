from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple, Type

from unstructured_inference.models.detectron2onnx import (
    MODEL_TYPES as DETECTRON2_ONNX_MODEL_TYPES,
)
from unstructured_inference.models.detectron2onnx import UnstructuredDetectronONNXModel
from unstructured_inference.models.unstructuredmodel import UnstructuredModel
from unstructured_inference.models.yolox import MODEL_TYPES as YOLOX_MODEL_TYPES
from unstructured_inference.models.yolox import UnstructuredYoloXModel
from unstructured_inference.utils import LazyDict

DEFAULT_MODEL = "yolox"

models: Dict[str, UnstructuredModel] = {}


def get_default_model_mappings() -> Tuple[
    Dict[str, Type[UnstructuredModel]],
    Dict[str, dict | LazyDict],
]:
    """default model mappings for models that are in `unstructured_inference` repo"""
    return {
        **{name: UnstructuredDetectronONNXModel for name in DETECTRON2_ONNX_MODEL_TYPES},
        **{name: UnstructuredYoloXModel for name in YOLOX_MODEL_TYPES},
    }, {**DETECTRON2_ONNX_MODEL_TYPES, **YOLOX_MODEL_TYPES}


model_class_map, model_config_map = get_default_model_mappings()


def register_new_model(model_config: dict, model_class: UnstructuredModel):
    """Register this model in model_config_map and model_class_map.

    Those maps are updated with the with the new model class information.
    """
    model_config_map.update(model_config)
    model_class_map.update({name: model_class for name in model_config})


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
        if model_name in model_config_map:
            initialize_params = model_config_map[model_name]
        else:
            raise UnknownModelException(f"Unknown model type: {model_name}")

    model: UnstructuredModel = model_class_map[model_name]()

    model.initialize(**initialize_params)
    models[model_name] = model
    return model


class UnknownModelException(Exception):
    """A model was requested with an unrecognized identifier."""

    pass
