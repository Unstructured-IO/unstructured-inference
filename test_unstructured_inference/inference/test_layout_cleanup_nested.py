"""Regression coverage for nested regions after a disjoint larger candidate."""

import numpy as np
import pytest

from unstructured_inference.inference.layoutelement import (
    LayoutElements,
    clean_layoutelements,
    clean_layoutelements_for_class,
    partition_groups_from_regions,
)
from unstructured_inference.models.unstructuredmodel import UnstructuredObjectDetectionModel


@pytest.mark.parametrize("cleanup", ["all", "class", "model"])
def test_cleanup_continues_after_disjoint_candidate(cleanup):
    coords = [[0, 0, 100, 100], [110, 0, 130, 20], [111, 1, 112, 2]]
    texts = ["unrelated", "outer", "duplicate"]
    classes = [0, 0, 0]
    if cleanup == "class":
        coords.append([200, 0, 201, 1])
        texts.append("other class")
        classes.append(1)
    elements = LayoutElements(
        element_coords=np.array(coords),
        texts=np.array(texts),
        element_class_ids=np.array(classes),
        element_class_id_map={0: "Text", 1: "Title"},
    )

    if cleanup == "class":
        result = clean_layoutelements_for_class(elements, element_class=0)
    elif cleanup == "model":
        assert [len(group) for group in partition_groups_from_regions(elements)] == [3]
        result = UnstructuredObjectDetectionModel.deduplicate_detected_elements(None, elements)
    else:
        result = clean_layoutelements(elements)

    expected = [0, 1, 3] if cleanup == "class" else [0, 1]
    np.testing.assert_array_equal(result.element_coords, elements.element_coords[expected])
    np.testing.assert_array_equal(result.texts, elements.texts[expected])
    np.testing.assert_array_equal(result.element_class_ids, elements.element_class_ids[expected])
