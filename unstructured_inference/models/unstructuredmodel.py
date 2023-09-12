from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
from PIL.Image import Image

from unstructured_inference.inference.elements import (
    Rectangle,
    intersect_free_quadrilaterals,
    intersections,
)

if TYPE_CHECKING:
    from unstructured_inference.inference.layoutelement import (
        LayoutElement,
        LocationlessLayoutElement,
    )


def draw(e1: LayoutElement, e2: LayoutElement):
    import PIL.Image
    from PIL import ImageDraw, ImageFont

    from unstructured_inference.inference.elements import minimal_containing_region

    try:
        kbd = ImageFont.truetype("Keyboard.ttf", 20)
        mini_page = minimal_containing_region(e1, e2)
        blank_page = PIL.Image.new("RGB", (2000, 2000))
        draw = ImageDraw.Draw(blank_page)

        draw.text((e1.x1, e1.y1), text=f"{e1.text[:7]}", fill="white", font=kbd)
        draw.rectangle((e1.x1, e1.y1, e1.x2, e1.y2), outline=(255, 0, 0))

        draw.text((e2.x1, e2.y1), text=f"{e2.text[:7]}", fill="white", font=kbd)
        draw.rectangle((e2.x1, e2.y1, e2.x2, e2.y2), outline=(0, 255, 0))
        blank_page.show()
    except:
        print("Ooops")


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

        from unstructured_inference.inference.elements import (
            grow_region_to_match_region,
            partition_groups_from_regions,
        )

        def check_rectangle(e: LayoutElement):
            well_formed = e.x1 < e.x2 and e.y1 < e.y2
            if not well_formed:
                print(f"Bad rectangle! {e.id}")

            return well_formed

        def probably_contained(
            element: Union[LayoutElement, Rectangle],
            inside: Union[LayoutElement, Rectangle],
            area_threshold: float = 0.2,
        ) -> bool:
            """This function checks if one element is inside other.
             If is definetly inside returns True, in other case check
            if the intersection is big enough."""
            if Rectangle.a_inside_b(element, inside):
                return True

            intersected_area = element.intersection(inside)
            if intersected_area:
                return intersected_area.area >= element.area * area_threshold
            return False

        def clean_tables(elements: List[LayoutElement]) -> Iterable[LayoutElement]:
            import numpy as np

            tables = [e for e in elements if e.type == "Table"]
            not_tables = [e for e in elements if e.type != "Table"]
            if len(tables) == 0:
                return elements

            nested_check = None
            for table in tables:
                nested_current = np.array([probably_contained(e, table) for e in not_tables])
                if nested_check is None:
                    nested_check = nested_current
                    continue
                nested_check = nested_check | nested_current

            final_elements = []
            for nested, elem in zip(nested_check, not_tables):  # type:ignore
                if not nested and elem not in [tables]:
                    final_elements.append(elem)

            final_elements.extend(tables)
            final_elements.sort(key=lambda e: e.y1)
            return final_elements

        def tag(elements):
            colors = ["red", "blue", "green", "cyan", "magenta", "brown"]

            max_x = 0
            max_y = 0
            for i, e in enumerate(elements):
                e.text = f"-{i}-:{e.text}"
                e.id = i
                e.clrs = colors[i % 6]
                max_x = max(max_x, e.x1)
                max_x = max(max_x, e.x2)
                max_y = max(max_y, e.y1)
                max_x = max(max_y, e.y2)

        def shrink_regions(elements: List[LayoutElement], jump: int = 0) -> List[LayoutElement]:
            intersections_mtx = intersections(*elements)

            for e in elements:
                check_rectangle(e)

            for i, row in enumerate(intersections_mtx):
                first = elements[i]
                if not first:
                    continue

                indices_to_check = np.where(row)[0]  # We get only the elements which intersected
                for j in indices_to_check:
                    if i != j and elements[j] is not None:
                        second = elements[j]
                        intersection = first.intersection(
                            second,
                        )  # we know it does, but need the region
                        first_inside_second = first.is_in(second)
                        second_inside_first = second.is_in(first)

                        if first_inside_second and not second_inside_first:
                            elements[i] = None
                        elif second_inside_first and not first_inside_second:
                            # delete second element
                            elements[j] = None
                        elif intersection:
                            iou = first.intersection_over_union(second)
                            if iou < 0.5:  # small
                                first, second = intersect_free_quadrilaterals(first, second)
                                if not check_rectangle(first) or first.height < 15:
                                    elements[i] = None  # The rectangle is too small, delete
                                if not check_rectangle(second) or second.height < 15:
                                    elements[j] = None

                            else:  # big
                                # merge
                                grow_region_to_match_region(first, second)
                                elements[j] = None

                if elements[i] is None:  # the element have been deleted,
                    continue

            elements = [e for e in elements if e is not None]
            return elements

        tag(elements)
        cleaned_elements: List[LayoutElement] = []
        print("Prechecking...")
        for e in elements:
            check_rectangle(e)
        print("Precheck done.")
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

        cleaned_elements = shrink_regions(cleaned_elements)
        # cleaned_elements = [e for e in cleaned_elements if e.height>15]
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
