class Column:
    def __init__(self, layout_elements: []):
        self.layout_elements = layout_elements

        num_elements = len(layout_elements)
        if num_elements > 0:
            self.x_midpoint = (
                sum([el.x_midpoint for el in layout_elements]) / num_elements
            )
        else:
            self.x_midpoint = 0

    def add_element(self, layout_element):
        self.layout_elements.append(layout_element)
        num_elements = len(self.layout_elements)
        self.x_midpoint = (
            sum([el.x_midpoint for el in self.layout_elements]) / num_elements
        )


def order_layout(layout):
    width = calculate_width(layout)

    column_tolerance = width / 4
    full_page_min_width = 0.9 * width

    layout.sort(key=lambda element: element.y1)

    sorted_layout = []
    columns = []
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


def sorted_layout_from_columns(columns):
    sorted_layout = []
    if len(columns) > 0:
        columns.sort(key=lambda column: column.x_midpoint)
        for column in columns:
            column.layout_elements.sort(key=lambda element: element.y1)
            for layout_element in column.layout_elements:
                sorted_layout.append(layout_element)
    return sorted_layout


def calculate_width(layout):
    min_x1 = min([element.x1 for element in layout])
    max_x2 = max([element.x2 for element in layout])

    return max_x2 - min_x1
