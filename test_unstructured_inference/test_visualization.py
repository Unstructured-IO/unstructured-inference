import pytest

from PIL import Image
import numpy as np

from unstructured_inference.inference.elements import Rectangle
from unstructured_inference.inference.layout import PageLayout
from unstructured_inference.visualize import draw_bbox, draw_yolox_bounding_boxes


@pytest.mark.parametrize(
    ("y_coords", "x_coords"),
    [
        (10, slice(10, 15)),
        (10, slice(16, 50)),
        (40, slice(1, 50)),
        (slice(10, 40), 1),
        (slice(10, 12), 50),
        (slice(14, 16), 50),
        (slice(19, 40), 50),
    ],
)
def test_visualize(y_coords, x_coords):
    test_image = np.ones((100, 100, 3))
    boxes = [[1, 10, 50, 40]]
    annotated_img = draw_yolox_bounding_boxes(
        test_image, boxes, scores=[0.8], cls_ids=[0], class_names=["thing"]
    )
    assert annotated_img[y_coords, x_coords, 0].sum() == 0.0


def test_draw_bbox():
    test_image_arr = np.ones((100, 100, 3), dtype="uint8")
    image = Image.fromarray(test_image_arr)
    page = PageLayout(number=1, image=image, layout=None)
    x1, y1, x2, y2 = (1, 10, 7, 11)
    page.elements = [Rectangle(x1, y1, x2, y2)]
    rect = Rectangle(x1, y1, x2, y2)
    annotated_image = draw_bbox(image=image, rect=rect)
    # annotated_image = page.annotate(colors=["red"])
    annotated_array = np.array(annotated_image)
    # Make sure the pixels on the edge of the box are red
    for i, expected in zip(range(3), [255, 0, 0]):
        assert all(annotated_array[y1, x1:x2, i] == expected)
        assert all(annotated_array[y2, x1:x2, i] == expected)
        assert all(annotated_array[y1:y2, x1, i] == expected)
        assert all(annotated_array[y1:y2, x2, i] == expected)
    # Make sure almost all the pixels are not changed
    assert ((annotated_array[:, :, 0] == 1).mean()) > 0.995
    assert ((annotated_array[:, :, 1] == 1).mean()) > 0.995
    assert ((annotated_array[:, :, 2] == 1).mean()) > 0.995
