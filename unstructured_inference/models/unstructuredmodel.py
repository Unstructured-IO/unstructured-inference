from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
from PIL.Image import Image

from unstructured_inference.inference.elements import Rectangle, intersections

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
        blank_page = PIL.Image.new("RGB", (int(mini_page.width) + 1, int(mini_page.height) + 1))
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
            if not (e.x1 < e.x2 and e.y1 < e.y2):
                print(f"Bad rectangle! {e.id}")

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

            final_elements.extend(tables)
            final_elements.sort(key=lambda e: e.y1)
            return final_elements

        def tag(elements):
            colors = ["red", "blue", "green", "cyan", "magenta", "brown"]

            max_x = 0
            max_y = 0
            for i, e in enumerate(elements):
                e.text = f"##{i}# -> {e.text}"
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
                            percentage_inside_first = intersection.area / first.area
                            percentage_inside_second = intersection.area / second.area

                            # If areas are similar enough
                            if 0.8 < first.area / second.area <= 1:
                                intersection_small = False
                                intersection_big = True
                                if intersection_small:
                                    # split
                                    pass
                                    continue

                                if intersection_big:
                                    # merge
                                    grow_region_to_match_region(first, second)
                                    elements[j] = None

                            if first.area > second.area:
                                elements[j] = None
                            if second.area > first.area:
                                elements[i] = None

                                try:
                                    check_rectangle(first)
                                    check_rectangle(second)
                                except:
                                    print("Rompiste un rectangulo!")

                if elements[i] is None:  # the element have been deleted
                    continue

            elements = [e for e in elements if e is not None]
            return elements

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

        tag(cleaned_elements)
        cleaned_elements = shrink_regions(cleaned_elements)
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
