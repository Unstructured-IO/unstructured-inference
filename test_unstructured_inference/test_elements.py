from random import randint
from unstructured_inference.inference import elements

rect1 = elements.Rectangle(0, 0, 10, 10)
rect2 = elements.Rectangle(20, 20, 30, 30)
rect3 = elements.Rectangle(2, 2, 30, 30)


def test_intersection():
    assert rect1.intersection(rect3) == elements.Rectangle(2, 2, 10, 10)


def intersect_brute(rect1, rect2):
    return any(
        (rect2.x1 <= x <= rect2.x2) and (rect2.y1 <= y <= rect2.y2)
        for x in range(rect1.x1, rect1.x2 + 1)
        for y in range(rect1.y1, rect1.y2 + 1)
    )


def rand_rect():
    x1 = randint(0, 20)
    y1 = randint(0, 20)
    return elements.Rectangle(x1, y1, x1 + 10, y1 + 10)


def rand_big_rect():
    x1 = randint(0, 10)
    y1 = randint(0, 10)
    return elements.Rectangle(x1, y1, x1 + 20, y1 + 20)


def test_intersects_overlap():
    for _ in range(1000):
        rect1 = rand_rect()
        rect2 = rand_rect()
        assert intersect_brute(rect1, rect2) == rect1.intersects(rect2) == rect2.intersects(rect1)


def test_intersects_subset():
    for _ in range(1000):
        rect1 = rand_rect()
        rect2 = rand_big_rect()
        assert intersect_brute(rect1, rect2) == rect1.intersects(rect2) == rect2.intersects(rect1)
