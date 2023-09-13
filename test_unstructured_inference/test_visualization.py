from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from unstructured_inference.inference.elements import Rectangle
from unstructured_inference.visualize import draw_bbox, draw_yolox_bounding_boxes, show_plot


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
        test_image,
        boxes,
        scores=[0.8],
        cls_ids=[0],
        class_names=["thing"],
    )
    assert annotated_img[y_coords, x_coords, 0].sum() == 0.0


def test_draw_bbox():
    test_image_arr = np.ones((100, 100, 3), dtype="uint8")
    image = Image.fromarray(test_image_arr)
    x1, y1, x2, y2 = (1, 10, 7, 11)
    rect = Rectangle(x1, y1, x2, y2)
    annotated_image = draw_bbox(image=image, rect=rect)
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


def test_show_plot_with_pil_image(mock_pil_image):
    mock_fig = MagicMock()
    mock_ax = MagicMock()

    with patch(
        "matplotlib.pyplot.subplots",
        return_value=(mock_fig, mock_ax),
    ) as mock_subplots, patch("matplotlib.pyplot.show") as mock_show, patch.object(
        mock_ax,
        "imshow",
    ) as mock_imshow:
        show_plot(mock_pil_image, desired_width=100)

    mock_subplots.assert_called()
    mock_imshow.assert_called_with(mock_pil_image)
    mock_show.assert_called()


def test_show_plot_with_numpy_image(mock_numpy_image):
    mock_fig = MagicMock()
    mock_ax = MagicMock()

    with patch(
        "matplotlib.pyplot.subplots",
        return_value=(mock_fig, mock_ax),
    ) as mock_subplots, patch("matplotlib.pyplot.show") as mock_show, patch.object(
        mock_ax,
        "imshow",
    ) as mock_imshow:
        show_plot(mock_numpy_image)

    mock_subplots.assert_called()
    mock_imshow.assert_called_with(mock_numpy_image)
    mock_show.assert_called()


def test_show_plot_with_unsupported_image_type():
    with pytest.raises(ValueError) as exec_info:
        show_plot("unsupported_image_type")

    assert "Unsupported Image Type" in str(exec_info.value)
