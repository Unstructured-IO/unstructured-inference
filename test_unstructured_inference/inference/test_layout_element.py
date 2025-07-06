from unstructured_inference.inference.layoutelement import LayoutElement, LayoutElements, TextRegion
import numpy as np


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


def test_layout_elements_iter_support():
    coords = np.array([[0, 0, 100, 100]])
    texts = np.array(["sample"])
    probs = np.array([0.9])
    class_ids = np.array([0])
    class_id_map = {0: "Text"}
    sources = np.array(["test_source"])
    text_as_html = np.array(["<p>sample</p>"])
    table_as_cells = np.array([None])

    layout_elements = LayoutElements(
        element_coords=coords,
        texts=texts,
        element_probs=probs,
        element_class_ids=class_ids,
        element_class_id_map=class_id_map,
        sources=sources,
        text_as_html=text_as_html,
        table_as_cells=table_as_cells,
    )

    # New feature test: __iter__() works
    elements = list(layout_elements)
    assert len(elements) == 1
    assert isinstance(elements[0], LayoutElement)
    assert elements[0].text == "sample"
    assert elements[0].type == "Text"
