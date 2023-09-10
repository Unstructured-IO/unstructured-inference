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
        from unstructured_inference.inference.elements import partition_groups_from_regions, grow_region_to_match_region
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
            final_elements.sort(key=lambda e: e.y1)
            return final_elements
        
        def tag(elements):
            colors = ["red","blue","green","cyan","magenta","brown"]

            max_x = 0
            max_y = 0 
            for i,e in enumerate(elements):
                e.text=f"##{i}# -> {e.text}"
                e.id = i
                e.clrs = colors[i%6]
                # e.x1=e.x1/10
                # e.x2=e.x2/10
                # e.y1=e.y1/10
                # e.y2=e.y2/10
                max_x=max(max_x,e.x1)
                max_x=max(max_x,e.x2)
                max_y=max(max_y,e.y1)
                max_x=max(max_y,e.y2)

            #print(f"X:{max_x}, Y:{max_y}")

        def shrink_regions(elements: List[LayoutElement],jump:int = 0)->List[LayoutElement]:
            final_elements = []
            i=0
            #tag(elements)

            aux_elements = []

            while len(elements)>=2+jump:
                #window = elements[i:i+2]
                first = elements.pop(0)
                second = elements[jump]
                if "##11" in first.text or "##11" in second.text or "##12" in first.text or "##12" in second.text:
                    a=1+1
                    pass

                first_inside_second = first.is_in(second)
                second_inside_first = second.is_in(first)
                ######################################################
                # HERE THE OVERLAPING IS COMPLETE, ONE IS INSIDE OTHER
                ######################################################
                    #print(":D")
                if first_inside_second and not second_inside_first:
                    # second fagocite first
                    grow_region_to_match_region(second,first)
                    elements.pop(0) # The element remains at top of the list, now is 
                                    # a bigger region and the first element is discarded
                    elements = [second] + elements
                elif second_inside_first and not first_inside_second:
                    # first fagocite second
                    grow_region_to_match_region(first,second)
                    elements.pop(0) # The element "second" remains at top of the list, we
                                    # need to remove it
                    elements = [first]+elements
                else:
                ######################################################
                # HERE THE OVERLAPING IS PARCIAL
                ######################################################
                    intersection = first.intersection(second)
                    if not intersection: 
                        # Yeiii, nothing to do, add first element
                        aux_elements.append(first)
                        continue
                    # If most the second element is inside first 
                    if intersection.area >= second.area* 0.8:
                        # First fagocite second as almost all area is inside first
                        grow_region_to_match_region(first,second)
                        elements.pop(0)
                        elements = [first]+elements
                    elif intersection.area >= first.area* 0.8:
                        # Second fagocite first 
                        grow_region_to_match_region(second,first)
                        elements.pop(0) # The element remains at top of the list, now is 
                                    # a bigger region and the first element is discarded
                        elements = [second] + elements
                    else:
                        #split the elements
                        # if the regions intersects: we shrink both regions to respect each 
                        # other limit, and just store the first as definitive (second could
                        # interact with further elements)
                        if "##11" in first.text or "##11" in second.text or "##12" in first.text or "##12" in second.text:
                            a=1+1
                        tmp_y = first.y2
                        first.y2 = second.y1-1
                        second.y1 = tmp_y+1
                        #second.y1=first.y2+1
                        #elements = [first] + elements
                        aux_elements.append(first)

            aux_elements.extend(elements)
            elements = aux_elements                    
                # i+=1

                # if final_elements == []:
                #     final_elements = aux_elements.copy()
                # elif len(aux_elements)<len(final_elements):
                #     final_elements=aux_elements
                #     print(f"{i} Len:{len(final_elements)}")
                # else:
                #     break

        
            return elements

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
        
        tag(cleaned_elements)
        cleaned_elements = shrink_regions(cleaned_elements)       
        cleaned_elements = shrink_regions(cleaned_elements,jump=1)       
        #cleaned_elements = shrink_regions(cleaned_elements,jump=2)       
        #cleaned_elements = shrink_regions(cleaned_elements,jump=3)      
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
