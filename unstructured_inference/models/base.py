from typing import Optional
from unstructured_inference.models.unstructuredmodel import UnstructuredModel

from unstructured_inference.models.detectron2 import (
    MODEL_TYPES as DETECTRON2_MODEL_TYPES,
    UnstructuredDetectronModel,
)
from unstructured_inference.models.yolox import (
    MODEL_TYPES as YOLOX_MODEL_TYPES,
    UnstructuredYoloXModel,
)


def get_model(model_name: Optional[str] = None) -> UnstructuredModel:
    """Gets the model object by model name."""
    # TODO(alan): These cases are similar enough that we can probably do them all together with
    # importlib
    if model_name in DETECTRON2_MODEL_TYPES:
        model: UnstructuredModel = UnstructuredDetectronModel()
        model.initialize(**DETECTRON2_MODEL_TYPES[model_name])
    elif model_name in YOLOX_MODEL_TYPES:
        model = UnstructuredYoloXModel()
        model.initialize(**YOLOX_MODEL_TYPES[model_name])
    else:
        raise UnknownModelException(f"Unknown model type: {model_name}")
    return model


class UnknownModelException(Exception):
    """Exception for the case where a model is called for with an unrecognized identifier."""

    pass
