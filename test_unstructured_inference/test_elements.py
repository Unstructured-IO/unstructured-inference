from random import randint
from unittest.mock import PropertyMock, patch

import pytest

from unstructured_inference.inference import elements


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


# TODO(alan): Make a better test layout
@pytest.fixture()
def sample_layout():
    return [
        elements.EmbeddedTextRegion(
            x1=453.00277777777774,
            y1=317.319341111111,
            x2=711.5338541666665,
            y2=358.28571222222206,
            text="LayoutParser:",
        ),
        elements.EmbeddedTextRegion(
            x1=726.4778125,
            y1=317.319341111111,
            x2=760.3308594444444,
            y2=357.1698966666667,
            text="A",
        ),
        elements.EmbeddedTextRegion(
            x1=775.2748177777777,
            y1=317.319341111111,
            x2=917.3579885555555,
            y2=357.1698966666667,
            text="Unified",
        ),
        elements.EmbeddedTextRegion(
            x1=932.3019468888888,
            y1=317.319341111111,
            x2=1071.8426522222221,
            y2=357.1698966666667,
            text="Toolkit",
        ),
        elements.EmbeddedTextRegion(
            x1=1086.7866105555556,
            y1=317.319341111111,
            x2=1141.2105142777777,
            y2=357.1698966666667,
            text="for",
        ),
        elements.EmbeddedTextRegion(
            x1=1156.154472611111,
            y1=317.319341111111,
            x2=1256.334784222222,
            y2=357.1698966666667,
            text="Deep",
        ),
        elements.EmbeddedTextRegion(
            x1=437.83888888888885,
            y1=367.13322999999986,
            x2=610.0171992222222,
            y2=406.9837855555556,
            text="Learning",
        ),
        elements.EmbeddedTextRegion(
            x1=624.9611575555555,
            y1=367.13322999999986,
            x2=741.6754646666665,
            y2=406.9837855555556,
            text="Based",
        ),
        elements.EmbeddedTextRegion(
            x1=756.619423,
            y1=367.13322999999986,
            x2=958.3867708333332,
            y2=406.9837855555556,
            text="Document",
        ),
        elements.EmbeddedTextRegion(
            x1=973.3307291666665,
            y1=367.13322999999986,
            x2=1092.0535042777776,
            y2=406.9837855555556,
            text="Image",
        ),
    ]


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


def test_partition_groups_from_regions(sample_layout):
    words = sample_layout
    groups = elements.partition_groups_from_regions(words)
    assert len(groups) == 1
    sorted_groups = sorted(groups, key=lambda group: group[0].y1)
    text = "".join([el.text for el in sorted_groups[-1]])
    assert text.startswith("Layout")


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
            assert rect.area() == width * height


def test_rectangle_iou():
    for _ in range(1000):
        rect1 = rand_rect()
        assert rect1.intersection_over_union(rect1) == 1.0
        rect2 = rand_rect(20)
        assert rect1.intersection_over_union(rect2) == rect2.intersection_over_union(rect1)
        if rect1.is_in(rect2):
            assert rect1.intersection_over_union(rect2) == rect1.area() / rect2.area()
        elif rect2.is_in(rect1):
            assert rect1.intersection_over_union(rect2) == rect2.area() / rect1.area()
        else:
            if rect1.intersection(rect2) is None:
                assert rect1.intersection_over_union(rect2) == 0.0
            else:
                intersection = rect1.intersection(rect2).area()
                assert rect1.intersection_over_union(rect2) == intersection / (
                    rect1.area() + rect2.area() - intersection
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
