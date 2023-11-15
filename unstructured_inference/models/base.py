from typing import Dict, Optional

from unstructured_inference.models.chipper import MODEL_TYPES as CHIPPER_MODEL_TYPES
from unstructured_inference.models.chipper import UnstructuredChipperModel
from unstructured_inference.models.detectron2 import (
    MODEL_TYPES as DETECTRON2_MODEL_TYPES,
)
from unstructured_inference.models.detectron2 import (
    UnstructuredDetectronModel,
)
from unstructured_inference.models.detectron2onnx import (
    MODEL_TYPES as DETECTRON2_ONNX_MODEL_TYPES,
)
from unstructured_inference.models.detectron2onnx import (
    UnstructuredDetectronONNXModel,
)
from unstructured_inference.models.super_gradients import (
    UnstructuredSuperGradients,
)
from unstructured_inference.models.unstructuredmodel import UnstructuredModel
from unstructured_inference.models.yolox import (
    MODEL_TYPES as YOLOX_MODEL_TYPES,
)
from unstructured_inference.models.yolox import (
    UnstructuredYoloXModel,
)

DEFAULT_MODEL = "yolox"

models: Dict[str, UnstructuredModel] = {}


def get_model(
    model_name: Optional[str] = None, 
    model_path: Optional[str] = None, 
    label_map: Optional[dict] = None, 
    input_shape: Optional[tuple] = None
    ) -> UnstructuredModel:
    """Gets the model object by model name."""
    # TODO(alan): These cases are similar enough that we can probably do them all together with
    # importlib

    global models

    if model_name is None:
        model_name = DEFAULT_MODEL

    if model_name in models:
        return models[model_name]

    if model_name in DETECTRON2_MODEL_TYPES:
        model: UnstructuredModel = UnstructuredDetectronModel()
        initialize_params = {**DETECTRON2_MODEL_TYPES[model_name]}
    elif model_name in DETECTRON2_ONNX_MODEL_TYPES:
        model = UnstructuredDetectronONNXModel()
        initialize_params = {**DETECTRON2_ONNX_MODEL_TYPES[model_name]}
    elif model_name in YOLOX_MODEL_TYPES:
        model = UnstructuredYoloXModel()
        initialize_params = {**YOLOX_MODEL_TYPES[model_name]}
    elif model_name in CHIPPER_MODEL_TYPES:
        model = UnstructuredChipperModel()
        initialize_params = {**CHIPPER_MODEL_TYPES[model_name]}
    elif model_name == "super_gradients":
        model = UnstructuredSuperGradients()
        initialize_params = {'model_path': model_path, 'label_map': label_map, 'input_shape': input_shape}
    else:
        raise UnknownModelException(f"Unknown model type: {model_name}")
    model.initialize(**initialize_params)
    models[model_name] = model
    return model


class UnknownModelException(Exception):
    """Exception for the case where a model is called for with an unrecognized identifier."""

    pass
