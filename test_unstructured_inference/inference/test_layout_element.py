import pytest
from layoutparser.elements import TextBlock
from layoutparser.elements.layout_elements import Rectangle as LPRectangle

from unstructured_inference.constants import SUBREGION_THRESHOLD_FOR_OCR, Source
from unstructured_inference.inference.elements import TextRegion
from unstructured_inference.inference.layoutelement import (
    LayoutElement,
    aggregate_ocr_text_by_block,
    get_elements_from_ocr_regions,
    merge_inferred_layout_with_ocr_layout,
    merge_text_regions,
    supplement_layout_with_ocr_elements,
)


def test_aggregate_ocr_text_by_block():
    expected = "A Unified Toolkit"
    ocr_layout = [
        TextRegion(0, 0, 20, 20, source="OCR", text="A"),
        TextRegion(50, 50, 150, 150, source="OCR", text="Unified"),
        TextRegion(150, 150, 300, 250, source="OCR", text="Toolkit"),
        TextRegion(200, 250, 300, 350, source="OCR", text="Deep"),
    ]
    region = TextRegion(0, 0, 250, 350, text="")

    text = aggregate_ocr_text_by_block(ocr_layout, region, 0.5)
    assert text == expected


def test_merge_text_regions(mock_embedded_text_regions):
    expected = TextRegion(
        x1=437.83888888888885,
        y1=317.319341111111,
        x2=1256.334784222222,
        y2=406.9837855555556,
        text="LayoutParser: A Unified Toolkit for Deep Learning Based Document Image",
    )

    merged_text_region = merge_text_regions(mock_embedded_text_regions)
    assert merged_text_region == expected


def test_get_elements_from_ocr_regions(mock_embedded_text_regions):
    expected = [
        LayoutElement(
            x1=437.83888888888885,
            y1=317.319341111111,
            x2=1256.334784222222,
            y2=406.9837855555556,
            text="LayoutParser: A Unified Toolkit for Deep Learning Based Document Image",
            type="UncategorizedText",
        ),
    ]

    elements = get_elements_from_ocr_regions(mock_embedded_text_regions)
    assert elements == expected


def test_supplement_layout_with_ocr_elements(mock_layout, mock_ocr_regions):
    ocr_elements = [
        LayoutElement(
            r.x1,
            r.y1,
            r.x2,
            r.y2,
            text=r.text,
            source=None,
            type="UncategorizedText",
        )
        for r in mock_ocr_regions
    ]

    final_layout = supplement_layout_with_ocr_elements(mock_layout, mock_ocr_regions)

    # Check if the final layout contains the original layout elements
    for element in mock_layout:
        assert element in final_layout

    # Check if the final layout contains the OCR-derived elements
    assert any(ocr_element in final_layout for ocr_element in ocr_elements)

    # Check if the OCR-derived elements that are subregions of layout elements are removed
    for element in mock_layout:
        for ocr_element in ocr_elements:
            if ocr_element.is_almost_subregion_of(element, SUBREGION_THRESHOLD_FOR_OCR):
                assert ocr_element not in final_layout


def test_merge_inferred_layout_with_ocr_layout(mock_inferred_layout, mock_ocr_regions):
    ocr_elements = [
        LayoutElement(
            r.x1,
            r.y1,
            r.x2,
            r.y2,
            text=r.text,
            source=None,
            type="UncategorizedText",
        )
        for r in mock_ocr_regions
    ]

    final_layout = merge_inferred_layout_with_ocr_layout(mock_inferred_layout, mock_ocr_regions)

    # Check if the inferred layout's text attribute is updated with aggregated OCR text
    assert final_layout[0].text == mock_ocr_regions[2].text

    # Check if the final layout contains both original elements and OCR-derived elements
    assert all(element in final_layout for element in mock_inferred_layout)
    assert any(element in final_layout for element in ocr_elements)


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
