import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from unstructured_inference.inference.elements import TextRegions, coords_intersections
from unstructured_inference.inference.layoutelement import (
    _intersection_component_labels,
    partition_groups_from_regions,
)


@pytest.mark.parametrize("count", [0, 1, 256, 257, 1000])
@pytest.mark.parametrize("seed", range(5))
def test_components_match_dense_graph(count, seed):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(-100, 100, (count, 2))
    coords = np.column_stack((starts, starts + rng.uniform(0, 12, (count, 2))))
    expected = connected_components(coords_intersections(coords))[1]
    np.testing.assert_array_equal(_intersection_component_labels(coords), expected)


@pytest.mark.parametrize("special", ["touching", "nested", "nan", "infinite", "inverted"])
def test_components_preserve_edge_case_semantics(special):
    coords = np.column_stack((np.arange(300), np.zeros(300), np.arange(300) + 1, np.ones(300)))
    if special == "nested":
        coords[0] = [-1, -1, 301, 2]
    elif special == "nan":
        coords[10, 0] = np.nan
    elif special == "infinite":
        coords[10, 2] = np.inf
    elif special == "inverted":
        coords[10] = [20, 1, 10, 0]
    np.testing.assert_array_equal(
        _intersection_component_labels(coords),
        connected_components(coords_intersections(coords))[1],
    )


def test_large_sparse_groups_preserve_input_order_without_dense_graph(monkeypatch):
    count = 2000
    x = np.repeat(np.arange(count // 2) * 20, 2)
    coords = np.column_stack((x, np.zeros(count), x + 1, np.ones(count)))
    order = np.random.default_rng(17).permutation(count)
    regions = TextRegions(element_coords=coords[order], texts=np.array(order).astype(str))

    def forbid_dense_graph(*args):
        pytest.fail("large inputs must not allocate the all-pairs graph")

    monkeypatch.setattr(
        "unstructured_inference.inference.layoutelement.coords_intersections", forbid_dense_graph
    )
    groups = partition_groups_from_regions(regions)
    expected = {}
    for index in order:
        expected.setdefault(index // 2, []).append(str(index))
    assert [group.texts.tolist() for group in groups] == list(expected.values())


def test_reversed_touching_chain_preserves_component_and_region_order():
    count = 2048
    x = np.arange(count, dtype=float)[::-1]
    coords = np.column_stack((x, np.zeros(count), x + 1, np.ones(count)))
    regions = TextRegions(element_coords=coords, texts=x.astype(str))

    np.testing.assert_array_equal(_intersection_component_labels(coords), np.zeros(count))
    groups = partition_groups_from_regions(regions)

    assert len(groups) == 1
    np.testing.assert_array_equal(groups[0].element_coords, regions.element_coords)
    np.testing.assert_array_equal(groups[0].texts, regions.texts)


def test_components_tolerate_coordinates_near_float_limits():
    coords = np.array([[-1e308, 0, -1e308, 0]] * 129 + [[1e308, 0, 1e308, 0]] * 128)
    with np.errstate(all="raise"):
        labels = _intersection_component_labels(coords)
    np.testing.assert_array_equal(labels, connected_components(coords_intersections(coords))[1])
