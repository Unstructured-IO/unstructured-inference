from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, List

import numpy as np
from PIL.Image import Image

from unstructured_inference.inference.elements import (
    grow_region_to_match_region,
    intersections,
    partition_groups_from_regions,
)
from unstructured_inference.inference.layoutelement import separate

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
    def deduplicate_detected_elements(
        elements: List[LayoutElement],
        min_text_size: int = 15,
    ) -> List[LayoutElement]:
        """Deletes overlapping elements in a list of elements. Also will delete elements
        of less than min_text_size pixels height"""

        if len(elements) <= 1:
            return elements

        def clean_tables(elements: List[LayoutElement]) -> Iterable[LayoutElement]:
            tables = [e for e in elements if e.type == "Table"]
            not_tables = [e for e in elements if e.type != "Table"]
            if len(tables) == 0:
                return elements

            nested_check = None
            for table in tables:
                nested_current = np.array([e.is_almost_subregion_of(table) for e in not_tables])
                if nested_check is None:
                    nested_check = nested_current
                    continue
                nested_check = nested_check | nested_current

            final_elements = []
            for nested, elem in zip(nested_check, not_tables):  # type:ignore
                if not nested:
                    final_elements.append(elem)

            final_elements.extend(tables)
            final_elements.sort(key=lambda e: e.y1)
            return final_elements

        def enhance_regions(
            elements: List[LayoutElement],
            min_text_size: int,
            iom_to_merge: float = 0.3,
        ) -> List[LayoutElement]:
            """This function traverses all the elements and either deletes nested elements,
            or merges or splits them depending on the iou score for both regions"""
            intersections_mtx = intersections(*elements)

            for i, row in enumerate(intersections_mtx):
                first = elements[i]
                if first:
                    # We get only the elements which intersected
                    indices_to_check = np.where(row)[0]
                    # Delete the first element, since it will always intersect with itself
                    indices_to_check = indices_to_check[indices_to_check != i]
                    if len(indices_to_check) == 0:
                        continue
                    if len(indices_to_check) > 1:  # sort by iom
                        iom_to_check = [
                            (j, first.intersection_over_minimum(elements[j]))
                            for j in indices_to_check
                            if elements[j] is not None
                        ]
                        iom_to_check.sort(
                            key=lambda x: x[1],
                            reverse=True,
                        )  # sort elements by iom, so we first check the greatest
                        indices_to_check = [x[0] for x in iom_to_check]  # type:ignore
                for j in indices_to_check:
                    if i != j and elements[j] is not None and elements[i] is not None:
                        second = elements[j]
                        intersection = first.intersection(
                            second,
                        )  # we know it does, but need the region
                        first_inside_second = first.is_in(second)
                        second_inside_first = second.is_in(first)

                        if first_inside_second and not second_inside_first:
                            elements[i] = None  # type:ignore
                        elif second_inside_first and not first_inside_second:
                            # delete second element
                            elements[j] = None  # type:ignore
                        elif intersection:
                            iom = first.intersection_over_minimum(second)
                            if iom < iom_to_merge:  # small
                                separate(first, second)
                                # The rectangle could become too small, which is a
                                # good size to delete?
                            else:  # big
                                # merge
                                if first.area > second.area:
                                    grow_region_to_match_region(first, second)
                                    elements[j] = None  # type:ignore
                                else:
                                    grow_region_to_match_region(second, first)
                                    elements[i] = None  # type:ignore

            elements = [e for e in elements if e is not None]
            return elements

        cleaned_elements: List[LayoutElement] = []
        # TODO: Delete nested elements with low or None probability
        # TODO: Keep most confident
        # TODO: Better to grow horizontally than vertically?
        groups = partition_groups_from_regions(elements)  # type:ignore
        for g in groups:
            g = clean_tables(g)  # type:ignore
            cleaned_elements.extend(g)  # type:ignore

        cleaned_elements = enhance_regions(cleaned_elements, min_text_size)
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
