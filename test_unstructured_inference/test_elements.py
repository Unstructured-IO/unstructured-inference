from random import randint
from unstructured_inference.inference import elements
from unstructured_inference.inference.layout import load_pdf


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


def test_intersects_overlap():
    for _ in range(1000):
        rect1 = rand_rect()
        rect2 = rand_rect()
        assert intersect_brute(rect1, rect2) == rect1.intersects(rect2) == rect2.intersects(rect1)


def test_intersects_subset():
    for _ in range(1000):
        rect1 = rand_rect()
        rect2 = rand_rect(20)
        assert intersect_brute(rect1, rect2) == rect1.intersects(rect2) == rect2.intersects(rect1)


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


def test_partition_groups_from_regions():
    words, _ = load_pdf("sample-docs/layout-parser-paper.pdf")
    groups = elements.partition_groups_from_regions(words[0])
    assert len(groups) == 9
    sorted_groups = sorted(groups, key=lambda group: group[0].y1)
    text = "".join([el.text for el in sorted_groups[-1]])
    assert text.startswith("Deep")
