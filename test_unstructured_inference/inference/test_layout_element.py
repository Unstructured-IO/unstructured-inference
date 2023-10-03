import pytest
from layoutparser.elements import TextBlock
from layoutparser.elements.layout_elements import Rectangle as LPRectangle

from unstructured_inference.constants import Source
from unstructured_inference.inference.layoutelement import (
    LayoutElement,
)


@pytest.mark.parametrize("is_table", [False, True])
def test_layout_element_extract_text(
    mock_layout_element,
    mock_text_region,
    mock_pil_image,
    is_table,
):
    if is_table:
        mock_layout_element.type = "Table"

    extracted_text = mock_layout_element.extract_text(
        objects=[mock_text_region],
        image=mock_pil_image,
        extract_tables=True,
    )

    assert isinstance(extracted_text, str)
    assert "Sample text" in extracted_text

    if mock_layout_element.type == "Table":
        assert hasattr(mock_layout_element, "text_as_html")


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
    expected = LayoutElement(100, 100, 300, 300, None, None)

    assert LayoutElement.from_region(mock_rectangle) == expected


def test_layout_element_from_lp_textblock():
    mock_text_block = TextBlock(
        block=LPRectangle(100, 100, 300, 300),
        text="Sample Text",
        type="Text",
        score=0.99,
    )

    expected = LayoutElement(
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
