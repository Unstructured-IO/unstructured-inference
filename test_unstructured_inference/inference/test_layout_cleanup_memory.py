"""Compatibility and allocation coverage for layout-region deduplication."""

import numpy as np
import pytest
from PIL import Image

from unstructured_inference.inference import layoutelement
from unstructured_inference.inference.layout import PageLayout
from unstructured_inference.inference.layoutelement import (
    EPSILON_AREA,
    LayoutElements,
    clean_layoutelements,
    intersection_areas_between_coords,
)
from unstructured_inference.models.unstructuredmodel import UnstructuredObjectDetectionModel


def _dense_cleanup_reference(elements, threshold):
    """All-pairs containment cleanup, used as an output-compatibility oracle."""
    if len(elements) < 2:
        return elements
    order = np.argsort(-elements.areas)
    coords = elements.element_coords[order]
    areas = elements.areas[order]
    with np.errstate(invalid="ignore", divide="ignore"):
        contained = (
            intersection_areas_between_coords(coords, coords) / np.maximum(areas, EPSILON_AREA)
            > threshold
        ) & (areas <= areas.T)
    count = len(elements)
    mask = np.ones_like(areas, dtype=bool)
    current = 0
    while count > 1:
        next_index = current + 1
        remove = np.flatnonzero(contained[current, next_index:]) + next_index
        if not remove.sum():
            break
        mask[remove] = False
        count -= len(remove) + 1
        remaining = np.flatnonzero(mask[next_index:])
        if not len(remaining):
            break
        current = remaining[0] + next_index
    indices = order[mask][np.argsort(coords[mask, 1])]
    return elements.slice(indices)


def _elements(coords):
    count = len(coords)
    labels = np.arange(count).astype(str)
    return LayoutElements(
        element_coords=coords,
        element_class_ids=np.arange(count) % 3,
        element_class_id_map={0: "Text", 1: "Table", 2: "Image"},
        element_probs=np.arange(count) / max(count, 1),
        texts=labels,
        sources=labels,
        is_extracted_array=np.arange(count) % 2,
        text_as_html=labels,
        table_as_cells=labels,
        table_extraction_method=labels,
    )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize(
    "case", ["random", "nested", "duplicate", "zero", "inverted", "nan", "inf", "overflow"]
)
def test_cleanup_matches_dense_output_and_all_attributes(seed, case):
    rng = np.random.default_rng(seed)
    count = [0, 1, 2, 8, 32, 128, 257, 300, 400, 500][seed]
    starts = rng.uniform(-10, 10, (count, 2))
    coords = np.column_stack((starts, starts + rng.uniform(0, 12, (count, 2))))
    if count:
        if case == "nested":
            coords[0] = [-100, -100, 100, 100]
        elif case == "duplicate":
            coords[::2] = coords[0]
        elif case == "zero":
            coords[::3, 2:] = coords[::3, :2]
        elif case == "inverted":
            coords[::3] = coords[::3, [2, 3, 0, 1]]
        elif case == "nan":
            coords[::3, seed % 4] = np.nan
        elif case == "inf":
            coords[::3, seed % 4] = np.inf if seed % 2 else -np.inf
        elif case == "overflow":
            coords *= 1e306
    regions = _elements(coords)
    for threshold in (-0.5, 0.0, 0.5, 1.0, np.nan):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            actual = clean_layoutelements(regions, threshold)
            expected = _dense_cleanup_reference(regions, threshold)
        for attr in ("element_coords", *regions._optional_array_attributes):
            np.testing.assert_array_equal(getattr(actual, attr), getattr(expected, attr))
        assert actual.element_class_id_map == expected.element_class_id_map


def test_large_connected_group_does_not_allocate_all_pairs(monkeypatch):
    count = 8000
    x = np.arange(count, dtype=float)
    regions = _elements(np.column_stack((x, np.zeros(count), x + 1, np.ones(count))))
    original = intersection_areas_between_coords
    calls = []

    def bounded_intersections(first, second, *args, **kwargs):
        assert len(first) == 1
        calls.append((len(first), len(second)))
        return original(first, second, *args, **kwargs)

    monkeypatch.setattr(layoutelement, "intersection_areas_between_coords", bounded_intersections)
    output = clean_layoutelements(regions)

    assert calls == [(1, count - 1)]
    assert len(output) == count
    assert set(output.texts) == set(regions.texts)


def test_detection_model_pipeline_preserves_connected_regions_and_routing(monkeypatch):
    count = 2000
    x = np.arange(count, dtype=float)
    regions = _elements(np.column_stack((x, np.zeros(count), x + 1, np.ones(count))))
    regions.routing = "table"
    regions.routing_score = 0.75

    class Detector(UnstructuredObjectDetectionModel):
        def initialize(self):
            pass

        def predict(self, image):
            return regions

    original = intersection_areas_between_coords

    def bounded_intersections(first, second, *args, **kwargs):
        assert len(first) == 1
        return original(first, second, *args, **kwargs)

    monkeypatch.setattr(layoutelement, "intersection_areas_between_coords", bounded_intersections)
    page = PageLayout.from_image(Image.new("RGB", (32, 32)), detection_model=Detector())
    expected = _dense_cleanup_reference(regions, 0.5)

    np.testing.assert_array_equal(page.elements_array.element_coords, expected.element_coords)
    np.testing.assert_array_equal(page.elements_array.texts, expected.texts)
    assert page.elements_array.routing == "table"
    assert page.elements_array.routing_score == 0.75
    assert page.image is None
