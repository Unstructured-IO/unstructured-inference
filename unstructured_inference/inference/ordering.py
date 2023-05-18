from typing import List

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
):
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
    width = calculate_width(layout)
    column_tolerance = column_tol_factor * width
    full_page_min_width = full_page_threshold_factor * width

    layout.sort(key=lambda element: element.y1)

    sorted_layout = []
    columns: List[Column] = []
    for layout_element in layout:
        if layout_element.width > full_page_min_width:
            sorted_layout.extend(sorted_layout_from_columns(columns))
            columns = []
            sorted_layout.append(layout_element)

        else:
            added_to_column = False
            for column in columns:
                difference = abs(layout_element.x_midpoint - column.x_midpoint)
                if difference < column_tolerance:
                    column.add_element(layout_element)
                    added_to_column = True
                    break

            if not added_to_column:
                columns.append(Column(layout_elements=[layout_element]))

    sorted_layout.extend(sorted_layout_from_columns(columns))
    return sorted_layout


def sorted_layout_from_columns(columns: List[Column]):
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


def calculate_width(layout):
    """Calculates total width of the elements in the layout. Used for computing the full
    page threshold and column tolerance."""
    min_x1 = min([element.x1 for element in layout])
    max_x2 = max([element.x2 for element in layout])

    return max_x2 - min_x1
