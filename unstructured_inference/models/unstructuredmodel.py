from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, List, Union

from PIL.Image import Image

from unstructured_inference.inference.elements import Rectangle

if TYPE_CHECKING:
    from unstructured_inference.inference.layoutelement import (
        LayoutElement,
        LocationlessLayoutElement,
    )


class UnstructuredModel(ABC):
    """Wrapper class for the various models used by unstructured."""

    def __init__(self):
        """model should support inference of some sort, either by calling or by some method.
        UnstructuredModel doesn't provide any training interface, it's assumed the model is
        already trained.
        """
        self.model = None

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Do inference using the wrapped model."""
        if self.model is None:
            raise ModelNotInitializedError(
                "Model has not been initialized. Please call the initialize method with the "
                "appropriate arguments for loading the model.",
            )
        pass  # pragma: no cover

    def __call__(self, x: Any) -> Any:
        """Inference using function call interface."""
        return self.predict(x)

    @abstractmethod
    def initialize(self, *args, **kwargs):
        """Load the model for inference."""
        pass  # pragma: no cover


class UnstructuredObjectDetectionModel(UnstructuredModel):
    """Wrapper class for object detection models used by unstructured."""

    @abstractmethod
    def predict(self, x: Image) -> List[LayoutElement]:
        """Do inference using the wrapped model."""
        super().predict(x)
        return []  # pragma: no cover

    def __call__(self, x: Image) -> List[LayoutElement]:
        """Inference using function call interface."""
        return super().__call__(x)

    @staticmethod
    def deduplicate_detected_elements(elements: List[LayoutElement]) -> List[LayoutElement]:
        """Deletes overlapping elements in a list of elements"""
        from unstructured_inference.inference.elements import partition_groups_from_regions

        def get_best_class(elements: List[LayoutElement]):
            import numpy as np

            index = np.argmax([e.prob if e.prob else 0 for e in elements])
            return elements[index].type, elements[index].prob

        def get_bigger_class(elements: List[LayoutElement]):
            import numpy as np

            index = np.argmax([e.area for e in elements])
            return elements[index].type, elements[index].prob

        def probably_contained(
            element: Union[LayoutElement, Rectangle],
            inside: Union[LayoutElement, Rectangle],
        ) -> bool:
            if Rectangle.a_inside_b(element, inside):
                return True

            intersected_area = element.intersection(inside)
            if intersected_area:
                return intersected_area.area >= element.area * 0.2
            return False

        def clean_tables(elements: List[LayoutElement]) -> Iterable[LayoutElement]:
            import numpy as np

            tables = [e for e in elements if e.type == "Table"]
            if len(tables) == 0:
                return elements

            nested = None
            for table in tables:
                nested_current = np.array([probably_contained(e, table) for e in elements])
                if nested is None:
                    nested = nested_current
                    continue
                nested = nested | nested_current

            final_elements = []
            for nested, elem in zip(nested, elements):  # type:ignore
                if not nested and elem not in [tables]:
                    final_elements.append(elem)
                # if elem.x1==727 and elem.y1==913: #Weird element not detected as intersected
                #     elem.type='GROUP'
                #     elem.intersection(tables[0])
                #     elem.intersection(tables[1])
                #     print(elem)

            final_elements.extend(tables)

            return final_elements

        cleaned_elements: List[LayoutElement] = []

        # TODO: Delete nested elements with low or None probability
        # TODO: Keep most confident
        # TODO: Better to grow horizontally than vertically?
        groups = partition_groups_from_regions(elements)  # type:ignore
        for g in groups:
            # group_border = minimal_containing_region(*g)
            # group_border.type="GROUP"              # JUST FOR DEBUGGING
            # group_border.prob = 1.0                # JUST FOR DEBUGGING
            # cleaned_elements.append(group_border)  # JUST FOR DEBUGGING

            g = clean_tables(g)  # type:ignore
            cleaned_elements.extend(g)  # type:ignore
        return cleaned_elements


class UnstructuredElementExtractionModel(UnstructuredModel):
    """Wrapper class for object extraction models used by unstructured."""

    @abstractmethod
    def predict(self, x: Image) -> List[LocationlessLayoutElement]:
        """Do inference using the wrapped model."""
        super().predict(x)
        return []  # pragma: no cover

    def __call__(self, x: Image) -> List[LocationlessLayoutElement]:
        """Inference using function call interface."""
        return super().__call__(x)


class ModelNotInitializedError(Exception):
    pass
