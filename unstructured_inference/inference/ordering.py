from typing import List

from unstructured_inference.inference.elements import TextRegion


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
