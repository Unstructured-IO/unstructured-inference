


from unstructured_inference.constants import Source
from unstructured_inference.inference.layoutelement import LayoutElement, TextRegion


def test_layout_element_do_dict(mock_layout_element):
    expected = {
        "coordinates": ((100, 100), (100, 300), (300, 300), (300, 100)),
        "text": "Sample text",
        "type": "Text",
        "prob": None,
        "source": None,
    }

    assert mock_layout_element.to_dict() == expected


def test_layout_element_from_region(mock_rectangle):
    expected = LayoutElement.from_coords(100, 100, 300, 300)
    region = TextRegion(bbox=mock_rectangle)

    assert LayoutElement.from_region(region) == expected


