from abc import ABC, abstractmethod
from typing import Any


class UnstructuredModel(ABC):
    """Wrapper class for the various models used by unstructured."""

    def __init__(self, model: Any):
        """model should support inference of some sort, either by calling or by some method.
        UnstructuredModel doesn't provide any training interface, it's assumed the model is
        already trained.
        """
        self.model = model

    @abstractmethod
    def __call__(self, x: Any) -> Any:
        pass  # pragma: no cover
