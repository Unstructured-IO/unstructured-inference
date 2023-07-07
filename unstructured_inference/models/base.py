from typing import Optional

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
from unstructured_inference.models.unstructuredmodel import UnstructuredModel
from unstructured_inference.models.yolox import (
    MODEL_TYPES as YOLOX_MODEL_TYPES,
)
from unstructured_inference.models.yolox import (
    UnstructuredYoloXModel,
)

DEFAULT_MODEL = "detectron2_onnx"


def get_model(model_name: Optional[str] = None) -> UnstructuredModel:
    """Gets the model object by model name."""
    # TODO(alan): These cases are similar enough that we can probably do them all together with
    # importlib
    if model_name is None:
        model_name = DEFAULT_MODEL

    if model_name in DETECTRON2_MODEL_TYPES:
        model: UnstructuredModel = UnstructuredDetectronModel()
        model.initialize(**DETECTRON2_MODEL_TYPES[model_name])
    elif model_name in DETECTRON2_ONNX_MODEL_TYPES:
        model = UnstructuredDetectronONNXModel()
        model.initialize(**DETECTRON2_ONNX_MODEL_TYPES[model_name])
    elif model_name in YOLOX_MODEL_TYPES:
        model = UnstructuredYoloXModel()
        model.initialize(**YOLOX_MODEL_TYPES[model_name])
    elif model_name in CHIPPER_MODEL_TYPES:
        model = UnstructuredChipperModel()
        model.initialize(**CHIPPER_MODEL_TYPES[model_name])
    else:
        raise UnknownModelException(f"Unknown model type: {model_name}")
    return model


class UnknownModelException(Exception):
    """Exception for the case where a model is called for with an unrecognized identifier."""

    pass
