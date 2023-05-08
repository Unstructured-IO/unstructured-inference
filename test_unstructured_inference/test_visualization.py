import pytest

import numpy as np

from unstructured_inference.visualize import draw_yolox_bounding_boxes


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
