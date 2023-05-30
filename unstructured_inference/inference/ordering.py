from typing import List, Union

from unstructured_inference.inference.elements import TextRegion


class Column:
    """Class to capture a column of text in the layout. Will update the midpoint of the
    column as layout elements are added to help with new element comparisons."""

    def __init__(self, layout_elements: List[TextRegion] = []):
        self.layout_elements = layout_elements

        num_elements = len(layout_elements)
        if num_elements > 0:
            self.x_midpoint = sum([el.x_midpoint for el in layout_elements]) / num_elements
        else:
            self.x_midpoint = 0

    def add_element(self, layout_element: TextRegion):
        """Adds an elements to the column and updates the midpoint."""
        self.layout_elements.append(layout_element)
        num_elements = len(self.layout_elements)
        self.x_midpoint = sum([el.x_midpoint for el in self.layout_elements]) / num_elements


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

    layout.sort(key=lambda element: element.y1)
    # NOTE(alan): Temporarily revert to orginal logic pending fixing the new logic
    # See code prior to this commit for new logic.
    return layout


def sorted_layout_from_columns(columns: List[Column]) -> List[TextRegion]:
    """Creates a sorted list of elements from a list of columns. Columns will be sorted
    left to right and elements within columns are sorted top to bottom."""
    sorted_layout = []
    if len(columns) > 0:
        columns.sort(key=lambda column: column.x_midpoint)
        for column in columns:
            column.layout_elements.sort(key=lambda element: element.y1)
            for layout_element in column.layout_elements:
                sorted_layout.append(layout_element)
    return sorted_layout


def calculate_width(layout) -> Union[float, int]:
    """Calculates total width of the elements in the layout. Used for computing the full
    page threshold and column tolerance."""
    min_x1 = min([element.x1 for element in layout])
    max_x2 = max([element.x2 for element in layout])

    return max_x2 - min_x1
