from typing import Optional
from unstructured_inference.models.unstructuredmodel import UnstructuredModel

from unstructured_inference.models.detectron2 import (
    MODEL_TYPES as DETECTRON2_MODEL_TYPES,
    UnstructuredDetectronModel,
)


def get_model(model_name: Optional[str] = None) -> UnstructuredModel:
    """Gets the model object by model name."""
    if model_name in DETECTRON2_MODEL_TYPES:
        model = UnstructuredDetectronModel()
        model.initialize(**DETECTRON2_MODEL_TYPES[model_name])
    else:
        raise UnknownModelException(f"Unknown model type: {model_name}")
    return model


class UnknownModelException(Exception):
    """Exception for the case where a model is called for with an unrecognized identifier."""

    pass
