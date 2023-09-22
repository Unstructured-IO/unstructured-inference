from typing import Dict, Optional

from unstructured_inference.logger import logger
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

DEFAULT_MODEL = "yolox_quantized"

models: Dict[str, UnstructuredModel] = {}


def get_model(model_name: Optional[str] = None) -> UnstructuredModel:
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
        model.initialize(**DETECTRON2_MODEL_TYPES[model_name])
    elif model_name in DETECTRON2_ONNX_MODEL_TYPES:
        model = UnstructuredDetectronONNXModel()
        model.initialize(**DETECTRON2_ONNX_MODEL_TYPES[model_name])
    elif model_name in YOLOX_MODEL_TYPES:
        model = UnstructuredYoloXModel()
        model.initialize(**YOLOX_MODEL_TYPES[model_name])
    elif model_name in CHIPPER_MODEL_TYPES:
        logger.warning(
            "The Chipper model is currently in Beta and is not yet ready for production use. "
            "You can reach out to the Unstructured engineering team in the Unstructured "
            "community Slack if you have any feedback on the Chipper model. "
            "You can join the community Slack here: "
            "https://join.slack.com/t/unstructuredw-kbe4326/shared_invite/"
            "zt-1x7cgo0pg-PTptXWylzPQF9xZolzCnwQ",
        )
        model = UnstructuredChipperModel()
        model.initialize(**CHIPPER_MODEL_TYPES[model_name])
    else:
        raise UnknownModelException(f"Unknown model type: {model_name}")
    models[model_name] = model
    return model


class UnknownModelException(Exception):
    """Exception for the case where a model is called for with an unrecognized identifier."""

    pass
