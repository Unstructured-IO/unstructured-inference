import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def mock_pil_image():
    return Image.new("RGB", (50, 50))


@pytest.fixture()
def mock_numpy_image():
    return np.zeros((50, 50, 3), np.uint8)
