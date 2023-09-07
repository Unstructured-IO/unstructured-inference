import numpy as np
import pytest
from PIL import Image

from unstructured_inference.inference.elements import EmbeddedTextRegion, Rectangle, TextRegion
from unstructured_inference.inference.layoutelement import LayoutElement


@pytest.fixture()
def mock_pil_image():
    return Image.new("RGB", (50, 50))


@pytest.fixture()
def mock_numpy_image():
    return np.zeros((50, 50, 3), np.uint8)


@pytest.fixture()
def mock_rectangle():
    return Rectangle(100, 100, 300, 300)


@pytest.fixture()
def mock_text_region():
    return TextRegion(100, 100, 300, 300, text="Sample text")


@pytest.fixture()
def mock_layout_element():
    return LayoutElement(100, 100, 300, 300, text="Sample text", type="Text")


@pytest.fixture()
def mock_embedded_text_regions():
    return [
        EmbeddedTextRegion(
            x1=453.00277777777774,
            y1=317.319341111111,
            x2=711.5338541666665,
            y2=358.28571222222206,
            text="LayoutParser:",
        ),
        EmbeddedTextRegion(
            x1=726.4778125,
            y1=317.319341111111,
            x2=760.3308594444444,
            y2=357.1698966666667,
            text="A",
        ),
        EmbeddedTextRegion(
            x1=775.2748177777777,
            y1=317.319341111111,
            x2=917.3579885555555,
            y2=357.1698966666667,
            text="Unified",
        ),
        EmbeddedTextRegion(
            x1=932.3019468888888,
            y1=317.319341111111,
            x2=1071.8426522222221,
            y2=357.1698966666667,
            text="Toolkit",
        ),
        EmbeddedTextRegion(
            x1=1086.7866105555556,
            y1=317.319341111111,
            x2=1141.2105142777777,
            y2=357.1698966666667,
            text="for",
        ),
        EmbeddedTextRegion(
            x1=1156.154472611111,
            y1=317.319341111111,
            x2=1256.334784222222,
            y2=357.1698966666667,
            text="Deep",
        ),
        EmbeddedTextRegion(
            x1=437.83888888888885,
            y1=367.13322999999986,
            x2=610.0171992222222,
            y2=406.9837855555556,
            text="Learning",
        ),
        EmbeddedTextRegion(
            x1=624.9611575555555,
            y1=367.13322999999986,
            x2=741.6754646666665,
            y2=406.9837855555556,
            text="Based",
        ),
        EmbeddedTextRegion(
            x1=756.619423,
            y1=367.13322999999986,
            x2=958.3867708333332,
            y2=406.9837855555556,
            text="Document",
        ),
        EmbeddedTextRegion(
            x1=973.3307291666665,
            y1=367.13322999999986,
            x2=1092.0535042777776,
            y2=406.9837855555556,
            text="Image",
        ),
    ]


@pytest.fixture()
def mock_ocr_regions():
    return [
        EmbeddedTextRegion(10, 10, 90, 90, "0"),
        EmbeddedTextRegion(200, 200, 300, 300, "1"),
        EmbeddedTextRegion(500, 320, 600, 350, "3"),
    ]


# TODO(alan): Make a better test layout
@pytest.fixture()
def mock_layout(mock_embedded_text_regions):
    return [
        LayoutElement(
            r.x1,
            r.y1,
            r.x2,
            r.y2,
            text=r.text,
            type="UncategorizedText",
        )
        for r in mock_embedded_text_regions
    ]


@pytest.fixture()
def mock_inferred_layout(mock_embedded_text_regions):
    return [
        LayoutElement(
            r.x1,
            r.y1,
            r.x2,
            r.y2,
            text=None,
            type="Text",
        )
        for r in mock_embedded_text_regions
    ]
