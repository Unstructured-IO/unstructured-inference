from layoutparser.elements import TextBlock
from layoutparser.elements.layout_elements import Rectangle as LPRectangle

from unstructured_inference.constants import Source
from unstructured_inference.inference.layoutelement import LayoutElement, TextRegion


def test_layout_element_extract_text(
    mock_layout_element,
    mock_text_region,
):
    extracted_text = mock_layout_element.extract_text(
        objects=[mock_text_region],
    )

    assert isinstance(extracted_text, str)
    assert "Sample text" in extracted_text


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


def test_layout_element_from_lp_textblock():
    mock_text_block = TextBlock(
        block=LPRectangle(100, 100, 300, 300),
        text="Sample Text",
        type="Text",
        score=0.99,
    )

    expected = LayoutElement.from_coords(
        100,
        100,
        300,
        300,
        text="Sample Text",
        source=Source.DETECTRON2_LP,
        type="Text",
        prob=0.99,
    )
    assert LayoutElement.from_lp_textblock(mock_text_block) == expected
