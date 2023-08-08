from typing import List, cast

from unstructured_inference.inference.elements import TextRegion
from unstructured_inference.inference.layout import DocumentLayout
from unstructured_inference.inference.layoutelement import LayoutElement


def order_layout(
    layout: List[TextRegion],
    column_tol_factor: float = 0.2,
    full_page_threshold_factor: float = 0.9,
) -> List[TextRegion]:
    """Orders the layout elements detected on a page. For groups of elements that are not
    the width of the page, the algorithm attempts to group elements into column based on
    the coordinates of the bounding box. Columns are ordered left to right, and elements
    within columns are ordered top to bottom.

    Parameters
    ----------
    layout
        the layout elements to order.
    column_tol_factor
        multiplied by the page width to find the tolerance for considering two elements as
        part of the same column.
    full_page_threshold_factor
        multiplied by the page width to find the minimum width an elements need to be
        for it to be considered a full page width element.
    """
    if len(layout) == 0:
        return []

    layout.sort(key=lambda element: (element.y1, element.x1, element.y2, element.x2))
    # NOTE(alan): Temporarily revert to orginal logic pending fixing the new logic
    # See code prior to this commit for new logic.
    return layout


def order_two_column_page(elements: List[LayoutElement], width_page: int) -> List[LayoutElement]:
    """Order the elements of a page with two columns."""
    # Split the image vertically
    vertical_line_x = width_page / 2

    # Determine the order of the bounding boxes
    left_boxes = []
    right_boxes = []
    both_sided_boxes = []

    new_bounding_boxes_ix = []
    for i, bbox in enumerate(elements):
        x_min = bbox.x1
        x_max = bbox.x2
        if x_min < vertical_line_x and x_max < vertical_line_x:
            left_boxes.append(bbox)
        elif x_min > vertical_line_x and x_max > vertical_line_x:
            right_boxes.append(bbox)
        else:
            both_sided_boxes.append(bbox)

    both_sided_boxes.sort(key=lambda box: box.x1)
    # Create new order
    new_bounding_boxes_ix.extend(both_sided_boxes)
    new_bounding_boxes_ix.extend(left_boxes)
    new_bounding_boxes_ix.extend(right_boxes)
    return new_bounding_boxes_ix


def order_two_column_document(document: DocumentLayout) -> DocumentLayout:
    """Orders all pages in the document assuming the existence of two columns"""
    for page in document.pages:
        bbox_elements = [el for el in page.elements if isinstance(el, LayoutElement)]
        no_bbox_elements = [el for el in page.elements if not isinstance(el, LayoutElement)]
        ordered_elements = order_two_column_page(
            cast(List[LayoutElement], bbox_elements),
            width_page=page.image_metadata["width"],
        )
        page.elements = ordered_elements + no_bbox_elements
    return document
