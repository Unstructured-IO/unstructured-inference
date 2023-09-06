from unstructured_inference.constants import SUBREGION_THRESHOLD_FOR_OCR
from unstructured_inference.inference.elements import TextRegion
from unstructured_inference.inference.layoutelement import (
    LayoutElement,
    aggregate_ocr_text_by_block,
    get_elements_from_ocr_regions,
    merge_text_regions,
    supplement_layout_with_ocr_elements,
)


def test_aggregate_ocr_text_by_block():
    expected = "A Unified Toolkit"
    ocr_layout = [
        TextRegion(0, 0, 20, 20, "A"),
        TextRegion(50, 50, 150, 150, "Unified"),
        TextRegion(150, 150, 300, 250, "Toolkit"),
        TextRegion(200, 250, 300, 350, "Deep"),
    ]
    region = TextRegion(0, 0, 250, 350, "")

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
