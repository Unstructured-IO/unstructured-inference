from unstructured_inference.inference.elements import TextRegion
from unstructured_inference.inference.layoutelement import aggregate_ocr_text_by_block


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
