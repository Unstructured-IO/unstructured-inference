import os
from random import randint
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest

from unstructured_inference.constants import ElementType
from unstructured_inference.inference import elements
from unstructured_inference.inference.elements import (
    ImageTextRegion,
    Rectangle,
    TextRegion,
    TextRegions,
)
from unstructured_inference.inference.layoutelement import (
    LayoutElement,
    LayoutElements,
    clean_layoutelements,
    clean_layoutelements_for_class,
    merge_inferred_layout_with_extracted_layout,
    partition_groups_from_regions,
    separate,
)

skip_outside_ci = os.getenv("CI", "").lower() in {"", "false", "f", "0"}


def intersect_brute(rect1, rect2):
    return any(
        (rect2.x1 <= x <= rect2.x2) and (rect2.y1 <= y <= rect2.y2)
        for x in range(rect1.x1, rect1.x2 + 1)
        for y in range(rect1.y1, rect1.y2 + 1)
    )


def rand_rect(size=10):
    x1 = randint(0, 30 - size)
    y1 = randint(0, 30 - size)
    return elements.Rectangle(x1, y1, x1 + size, y1 + size)


@pytest.fixture()
def test_layoutelements():
    coords = np.array(
        [
            [0.6, 0.6, 0.65, 0.65],  # One little table nested inside all the others
            [0.5, 0.5, 0.7, 0.7],  # One nested table
            [0, 0, 1, 1],  # Big table
            [0.01, 0.01, 1.01, 1.01],
            [0.02, 0.02, 1.02, 1.02],
            [0.03, 0.03, 1.03, 1.03],
            [0.04, 0.04, 1.04, 1.04],
            [0.05, 0.05, 1.05, 1.05],
            [2, 2, 3, 3],  # Big table
        ],
    )
    element_class_ids = np.array([1, 1, 1, 0, 0, 0, 0, 0, 2])
    class_map = {0: "type0", 1: "type1", 2: "type2"}
    return LayoutElements(
        element_coords=coords,
        element_class_ids=element_class_ids,
        element_class_id_map=class_map,
    )


@pytest.mark.parametrize(
    ("rect1", "rect2", "expected"),
    [
        (Rectangle(0, 0, 1, 1), Rectangle(0, 0, None, None), None),
        (Rectangle(0, 0, None, None), Rectangle(0, 0, 1, 1), None),
    ],
)
def test_unhappy_intersection(rect1, rect2, expected):
    assert rect1.intersection(rect2) == expected
    assert not rect1.intersects(rect2)


@pytest.mark.parametrize("second_size", [10, 20])
def test_intersects(second_size):
    for _ in range(1000):
        rect1 = rand_rect()
        rect2 = rand_rect(second_size)
        assert intersect_brute(rect1, rect2) == rect1.intersects(rect2) == rect2.intersects(rect1)
        if rect1.intersects(rect2):
            if rect1.is_in(rect2):
                assert rect1.intersection(rect2) == rect1 == rect2.intersection(rect1)
            elif rect2.is_in(rect1):
                assert rect2.intersection(rect1) == rect2
            else:
                x1 = max(rect1.x1, rect2.x1)
                x2 = min(rect1.x2, rect2.x2)
                y1 = max(rect1.y1, rect2.y1)
                y2 = min(rect1.y2, rect2.y2)
                intersection = elements.Rectangle(x1, y1, x2, y2)
                assert rect1.intersection(rect2) == intersection == rect2.intersection(rect1)
        else:
            assert rect1.intersection(rect2) is None
            assert rect2.intersection(rect1) is None


def test_intersection_of_lots_of_rects():
    for _ in range(1000):
        n_rects = 10
        rects = [rand_rect(6) for _ in range(n_rects)]
        intersection_mtx = elements.intersections(*rects)
        for i in range(n_rects):
            for j in range(n_rects):
                assert (
                    intersect_brute(rects[i], rects[j])
                    == intersection_mtx[i, j]
                    == intersection_mtx[j, i]
                )


def test_rectangle_width_height():
    for _ in range(1000):
        x1 = randint(0, 50)
        x2 = randint(x1 + 1, 100)
        y1 = randint(0, 50)
        y2 = randint(y1 + 1, 100)
        rect = elements.Rectangle(x1, y1, x2, y2)
        assert rect.width == x2 - x1
        assert rect.height == y2 - y1


def test_minimal_containing_rect():
    for _ in range(1000):
        rect1 = rand_rect()
        rect2 = rand_rect()
        big_rect = elements.minimal_containing_region(rect1, rect2)
        for decrease_attr in ["x1", "y1", "x2", "y2"]:
            almost_as_big_rect = rand_rect()
            mod = 1 if decrease_attr.endswith("1") else -1
            for attr in ["x1", "y1", "x2", "y2"]:
                if attr == decrease_attr:
                    setattr(almost_as_big_rect, attr, getattr(big_rect, attr) + mod)
                else:
                    setattr(almost_as_big_rect, attr, getattr(big_rect, attr))
            assert not rect1.is_in(almost_as_big_rect) or not rect2.is_in(almost_as_big_rect)

        assert rect1.is_in(big_rect)
        assert rect2.is_in(big_rect)


def test_partition_groups_from_regions(mock_embedded_text_regions):
    words = TextRegions.from_list(mock_embedded_text_regions)
    groups = partition_groups_from_regions(words)
    assert len(groups) == 1
    text = "".join(groups[-1].texts)
    assert text.startswith("Layout")
    # test backward compatibility
    text = "".join([str(region) for region in groups[-1].as_list()])
    assert text.startswith("Layout")


def test_rectangle_padding():
    rect = Rectangle(x1=0, y1=1, x2=3, y2=4)
    padded = rect.pad(1)
    assert (padded.x1, padded.y1, padded.x2, padded.y2) == (-1, 0, 4, 5)
    assert (rect.x1, rect.y1, rect.x2, rect.y2) == (0, 1, 3, 4)


def test_rectangle_area(monkeypatch):
    for _ in range(1000):
        width = randint(0, 20)
        height = randint(0, 20)
        with patch(
            "unstructured_inference.inference.elements.Rectangle.height",
            new_callable=PropertyMock,
        ) as mockheight, patch(
            "unstructured_inference.inference.elements.Rectangle.width",
            new_callable=PropertyMock,
        ) as mockwidth:
            rect = elements.Rectangle(0, 0, 0, 0)
            mockheight.return_value = height
            mockwidth.return_value = width
            assert rect.area == width * height


def test_rectangle_iou():
    for _ in range(1000):
        rect1 = rand_rect()
        assert rect1.intersection_over_union(rect1) == 1.0
        rect2 = rand_rect(20)
        assert rect1.intersection_over_union(rect2) == rect2.intersection_over_union(rect1)
        if rect1.is_in(rect2):
            assert rect1.intersection_over_union(rect2) == rect1.area / rect2.area
        elif rect2.is_in(rect1):
            assert rect1.intersection_over_union(rect2) == rect2.area / rect1.area
        else:
            if rect1.intersection(rect2) is None:
                assert rect1.intersection_over_union(rect2) == 0.0
            else:
                intersection = rect1.intersection(rect2).area
                assert rect1.intersection_over_union(rect2) == intersection / (
                    rect1.area + rect2.area - intersection
                )


def test_midpoints():
    for _ in range(1000):
        x2 = randint(0, 100)
        y2 = randint(0, 100)
        rect1 = elements.Rectangle(0, 0, x2, y2)
        assert rect1.x_midpoint == x2 / 2.0
        assert rect1.y_midpoint == y2 / 2.0
        x_offset = randint(0, 50)
        y_offset = randint(0, 50)
        rect2 = elements.Rectangle(x_offset, y_offset, x2 + x_offset, y2 + y_offset)
        assert rect2.x_midpoint == (x2 / 2.0) + x_offset
        assert rect2.y_midpoint == (y2 / 2.0) + y_offset


def test_is_disjoint():
    for _ in range(1000):
        a = randint(0, 100)
        b = randint(a + 1, 200)
        c = randint(b + 1, 300)
        d = randint(c + 1, 400)
        e = randint(0, 100)
        f = randint(e, 200)
        g = randint(0, 100)
        h = randint(g, 200)
        rect1 = elements.Rectangle(a, e, b, f)
        rect2 = elements.Rectangle(c, g, d, h)
        assert rect1.is_disjoint(rect2)
        assert rect2.is_disjoint(rect1)
        rect3 = elements.Rectangle(e, a, f, b)
        rect4 = elements.Rectangle(g, c, h, d)
        assert rect3.is_disjoint(rect4)
        assert rect4.is_disjoint(rect3)


@pytest.mark.parametrize(
    ("rect1", "rect2", "expected"),
    [
        (elements.Rectangle(0, 0, 100, 200), elements.Rectangle(0, 0, 60, 150), 1.0),
        (elements.Rectangle(0, 0, 100, 100), elements.Rectangle(150, 150, 200, 200), 0.0),
        (elements.Rectangle(0, 0, 100, 100), elements.Rectangle(50, 50, 150, 150), 0.25),
        (elements.Rectangle(0, 0, 100, 100), elements.Rectangle(20, 20, 120, 40), 0.8),
    ],
)
def test_intersection_over_min(
    rect1: elements.Rectangle,
    rect2: elements.Rectangle,
    expected: float,
):
    assert (
        rect1.intersection_over_minimum(rect2) == rect2.intersection_over_minimum(rect1) == expected
    )


def test_grow_region_to_match_region():
    from unstructured_inference.inference.elements import (
        Rectangle,
        grow_region_to_match_region,
    )

    a = Rectangle(1, 1, 2, 2)
    b = Rectangle(1, 1, 5, 5)
    grow_region_to_match_region(a, b)
    assert a == Rectangle(1, 1, 5, 5)


@pytest.mark.parametrize(
    ("rect1", "rect2", "expected"),
    [
        (elements.Rectangle(0, 0, 5, 5), elements.Rectangle(3, 3, 5.1, 5.1), True),
        (elements.Rectangle(0, 0, 5, 5), elements.Rectangle(3, 3, 5.2, 5.2), True),
        (elements.Rectangle(0, 0, 5, 5), elements.Rectangle(7, 7, 10, 10), False),
    ],
)
def test_is_almost_subregion_of(rect1, rect2, expected):
    assert expected == rect2.is_almost_subregion_of(rect1)


@pytest.mark.parametrize(
    ("rect1", "rect2"),
    [
        (elements.Rectangle(0, 0, 5, 5), elements.Rectangle(3, 3, 6, 6)),
        (elements.Rectangle(0, 0, 5, 5), elements.Rectangle(6, 6, 8, 8)),
        (elements.Rectangle(3, 3, 7, 7), elements.Rectangle(2, 2, 4, 4)),
        (elements.Rectangle(2, 2, 4, 11), elements.Rectangle(3, 3, 7, 10)),
        (elements.Rectangle(2, 2, 4, 4), elements.Rectangle(3, 3, 7, 10)),
        (elements.Rectangle(2, 2, 4, 4), elements.Rectangle(2.5, 2.5, 3.5, 4.5)),
        (elements.Rectangle(2, 2, 4, 4), elements.Rectangle(3, 1, 4, 3.5)),
        (elements.Rectangle(2, 2, 4, 4), elements.Rectangle(3, 1, 4.5, 3.5)),
    ],
)
def test_separate(rect1, rect2):
    separate(rect1, rect2)

    # assert not rect1.intersects(rect2) #TODO: fix this test


def test_merge_inferred_layout_with_extracted_layout():
    inferred_layout = [
        LayoutElement.from_coords(453, 322, 1258, 408, text=None, type=ElementType.SECTION_HEADER),
        LayoutElement.from_coords(387, 477, 1320, 537, text=None, type=ElementType.TEXT),
    ]

    extracted_layout = [
        TextRegion.from_coords(438, 318, 1272, 407, text="Example Section Header"),
        TextRegion.from_coords(377, 469, 1335, 535, text="Example Title"),
    ]

    extracted_layout_with_full_page_image = [
        ImageTextRegion.from_coords(0, 0, 1700, 2200, text="Example Section Header"),
    ]

    merged_layout = merge_inferred_layout_with_extracted_layout(
        inferred_layout=inferred_layout,
        extracted_layout=extracted_layout,
        page_image_size=(1700, 2200),
    )
    assert merged_layout[0].type == ElementType.SECTION_HEADER
    assert merged_layout[0].text == "Example Section Header"
    assert merged_layout[1].type == ElementType.TEXT
    assert merged_layout[1].text == "Example Title"

    # case: extracted layout with a full page image
    merged_layout = merge_inferred_layout_with_extracted_layout(
        inferred_layout=inferred_layout,
        extracted_layout=extracted_layout_with_full_page_image,
        page_image_size=(1700, 2200),
    )
    assert merged_layout == inferred_layout


def test_clean_layoutelements(test_layoutelements):
    elements = clean_layoutelements(test_layoutelements).as_list()
    assert len(elements) == 2
    assert (
        elements[0].bbox.x1,
        elements[0].bbox.y1,
        elements[0].bbox.x2,
        elements[0].bbox.x2,
    ) == (0, 0, 1, 1)
    assert (
        elements[1].bbox.x1,
        elements[1].bbox.y1,
        elements[1].bbox.x2,
        elements[1].bbox.x2,
    ) == (2, 2, 3, 3)


@pytest.mark.parametrize(
    ("coords", "class_ids", "expected_coords", "expected_ids"),
    [
        ([[0, 0, 1, 1], [0, 0, 1, 1]], [0, 1], [[0, 0, 1, 1]], [0]),  # one box
        (
            [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 2, 2]],
            [0, 1, 0],
            [[0, 0, 1, 1], [1, 1, 2, 2]],
            [0, 0],
        ),
        (
            [[0, 0, 1.4, 1.4], [0, 0, 1, 1], [0.4, 0, 1.4, 1], [1.2, 0, 1.4, 1]],
            [0, 1, 1, 1],
            [[0, 0, 1.4, 1.4]],
            [0],
        ),
    ],
)
def test_clean_layoutelements_cases(
    coords,
    class_ids,
    expected_coords,
    expected_ids,
):
    coords = np.array(coords)
    element_class_ids = np.array(class_ids)
    elements = LayoutElements(element_coords=coords, element_class_ids=element_class_ids)

    elements = clean_layoutelements(elements)
    np.testing.assert_array_equal(elements.element_coords, expected_coords)
    np.testing.assert_array_equal(elements.element_class_ids, expected_ids)


@pytest.mark.parametrize(
    ("coords", "class_ids", "class_to_filter", "expected_coords", "expected_ids"),
    [
        ([[0, 0, 1, 1], [0, 0, 1, 1]], [0, 1], 1, [[0, 0, 1, 1]], [1]),  # one box
        (
            [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 2, 2]],  # one box
            [0, 1, 0],
            1,
            [[0, 0, 1, 1], [1, 1, 2, 2]],
            [1, 0],
        ),
        (
            # a -> b, b -> c, but a not -> c
            [[0, 0, 1.4, 1.4], [0, 0, 1, 1], [0.4, 0, 1.4, 1], [1.2, 0, 1.4, 1]],
            [0, 1, 1, 1],
            1,
            [[0, 0, 1, 1], [1.2, 0, 1.4, 1], [0, 0, 1.4, 1.4]],
            [1, 1, 0],
        ),
        (
            # like the case above but a different filtering element type changes the results
            [[0, 0, 1.4, 1.4], [0, 0, 1, 1], [0.4, 0, 1.4, 1], [1.2, 0, 1.4, 1]],
            [0, 1, 1, 1],
            0,
            [[0, 0, 1.4, 1.4]],
            [0],
        ),
    ],
)
def test_clean_layoutelements_for_class(
    coords,
    class_ids,
    class_to_filter,
    expected_coords,
    expected_ids,
):
    coords = np.array(coords)
    element_class_ids = np.array(class_ids)
    elements = LayoutElements(element_coords=coords, element_class_ids=element_class_ids)

    elements = clean_layoutelements_for_class(elements, element_class=class_to_filter)
    np.testing.assert_array_equal(elements.element_coords, expected_coords)
    np.testing.assert_array_equal(elements.element_class_ids, expected_ids)
