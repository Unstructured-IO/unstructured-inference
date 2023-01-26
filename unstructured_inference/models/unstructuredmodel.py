from abc import ABC, abstractmethod
from typing import Any
from PIL import Image


class UnstructuredModel(ABC):
    """Wrapper class for the various models used by unstructured."""

    def __init__(self, model: Any):
        self.model = model

    @abstractmethod
    def __call__(self, x: Image) -> Any:
        pass
