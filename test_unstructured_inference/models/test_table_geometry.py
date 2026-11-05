"""Output equivalence, stable assignment and unusual rectangle semantics."""

import json
import random

import numpy as np
import pytest

from unstructured_inference.models import table_postprocess as pp
from unstructured_inference.models.tables import compute_confidence_score


def objects(boxes):
    return [{"bbox": box} for box in boxes]


@pytest.mark.parametrize("threshold", [0, 0.5, np.nextafter(0.5, 1), 1, 1.1])
@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("coordinate_type", [float, np.float64])
def test_geometry_matches_scalar_with_chunking(monkeypatch, threshold, forced, coordinate_type):
    rng = random.Random(20261105)

    def random_boxes(count):
        result = []
        for _ in range(count):
            x, y = rng.uniform(-500, 500), rng.uniform(-500, 500)
            result.append(
                [
                    coordinate_type(value)
                    for value in [x, y, x + rng.uniform(0.01, 500), y + rng.uniform(0.01, 500)]
                ],
            )
        return objects(result)

    containers = random_boxes(11)
    containers.extend(containers[:2])  # Equal-score containers retain their original order.
    packages = random_boxes(31)
    monkeypatch.setattr(pp, "_MAX_GEOMETRY_PAIRS", 39)  # Exercise many package chunks.
    actual_assignment = pp.slot_into_containers(containers, packages, threshold, forced)
    expected_assignment = pp._slot_into_containers_scalar(containers, packages, threshold, forced)
    assert actual_assignment == expected_assignment
    actual_scores, expected_scores = actual_assignment[2], expected_assignment[2]
    assert [type(score) for score in actual_scores] == [type(score) for score in expected_scores]
    assert compute_confidence_score(actual_scores) == compute_confidence_score(expected_scores)
    expected = pp._get_bbox_span_subset_scalar(packages, containers[0]["bbox"], threshold)
    actual = pp.get_bbox_span_subset(packages, containers[0]["bbox"], threshold)
    assert len(actual) == len(expected)
    assert all(a is b for a, b in zip(actual, expected))


def test_ties_thresholds_and_package_order():
    containers = objects([[0, 0, 1, 1], [0, 0, 1, 1]])
    packages = objects([[0, 0, 2, 1], [0, 0, 2, 1], [3, 3, 4, 4]])
    assert pp.slot_into_containers(containers, packages) == (
        [[0, 1], []],
        [[0], [0], []],
        [0.5, 0.5, 0.0],
    )
    assert pp.slot_into_containers(containers, packages, np.nextafter(0.5, 1)) == (
        [[], []],
        [[], [], []],
        [0.5, 0.5, 0.0],
    )
    assert pp.slot_into_containers(containers, packages, forced_assignment=True) == (
        [[0, 1, 2], []],
        [[0], [0], [0]],
        [0.5, 0.5, 0.0],
    )


@pytest.mark.parametrize(
    "box",
    [
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, float("nan"), 1],
        [0, 0, float("inf"), 1],
        [0, 0, 2**54, 1],
        [0, 0, 1e-200, 1e-200],
        [np.float32(0), 0, 1, 1],
    ],
)
def test_unsupported_geometry_preserves_legacy_behavior(box):
    containers = objects([box, [0, 0, 1, 1]])
    packages = objects([[0, 0, 2, 1], box])
    # JSON also compares NaNs in the legacy outputs without NaN != NaN surprises.
    assert json.dumps(pp.slot_into_containers(containers, packages), default=float) == json.dumps(
        pp._slot_into_containers_scalar(containers, packages), default=float
    )
    actual = pp.get_bbox_span_subset(packages, box)
    expected = pp._get_bbox_span_subset_scalar(packages, box)
    assert len(actual) == len(expected)
    assert all(a is b for a, b in zip(actual, expected))


def test_empty_container_has_legacy_intersection_semantics():
    # Rect.intersect replaces an empty left rectangle with the right rectangle.
    assert pp.slot_into_containers(objects([[0, 0, 0, 0]]), objects([[1, 1, 2, 2]])) == (
        [[0]],
        [[0]],
        [1.0],
    )


def test_empty_inputs():
    assert pp.slot_into_containers([], objects([[0, 0, 1, 1]])) == ([], [[]], [])
    assert pp.slot_into_containers(objects([[0, 0, 1, 1]]), []) == ([[]], [], [])
    assert pp.get_bbox_span_subset([], [0, 0, 1, 1]) == []


def test_container_count_over_pair_limit(monkeypatch):
    monkeypatch.setattr(pp, "_MAX_GEOMETRY_PAIRS", 1)
    containers = objects([[0, 0, 1, 1], [0, 0, 1, 1]])
    packages = objects([[0, 0, 1, 1]])
    assert pp.slot_into_containers(containers, packages) == ([[0], []], [[0]], [1.0])


def test_float64_coordinates_use_fast_path(mocker):
    containers = objects([[0, 0, 10, 10], [0, 0, 10, 10]])
    packages = objects(np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64))
    expected = pp._slot_into_containers_scalar(containers, packages)
    expected_subset = pp._get_bbox_span_subset_scalar(packages, containers[0]["bbox"])
    assignment_fallback = mocker.patch.object(
        pp,
        "_slot_into_containers_scalar",
        side_effect=AssertionError("unexpected scalar fallback"),
    )
    subset_fallback = mocker.patch.object(
        pp,
        "_get_bbox_span_subset_scalar",
        side_effect=AssertionError("unexpected scalar fallback"),
    )
    assert pp.slot_into_containers(containers, packages) == expected
    actual_subset = pp.get_bbox_span_subset(packages, containers[0]["bbox"])
    assert all(actual is expected for actual, expected in zip(actual_subset, expected_subset))
    assert len(actual_subset) == len(expected_subset)
    assignment_fallback.assert_not_called()
    subset_fallback.assert_not_called()


@pytest.mark.parametrize("count", [0, 1, 2])
def test_numpy_object_collections(count):
    containers = np.array(objects([[0, 0, 10, 10]]), dtype=object)
    packages = np.array(objects([[1, 1, 2, 2], [3, 3, 4, 4]])[:count], dtype=object)
    assert pp.slot_into_containers(containers, packages) == pp._slot_into_containers_scalar(
        containers,
        packages,
    )
    actual = pp.get_bbox_span_subset(packages, containers[0]["bbox"])
    expected = pp._get_bbox_span_subset_scalar(packages, containers[0]["bbox"])
    assert len(actual) == len(expected)
    assert all(a is b for a, b in zip(actual, expected))


@pytest.mark.parametrize("wrap", [tuple, iter])
def test_span_iterables(wrap):
    spans = objects([[0, 0, 1, 1], [3, 3, 4, 4]])
    actual = pp.get_bbox_span_subset(wrap(spans), [0, 0, 2, 2])
    assert len(actual) == 1
    assert actual[0] is spans[0]


@pytest.mark.parametrize("container_type", [float, np.float64])
@pytest.mark.parametrize("package_type", [float, np.float64])
def test_score_types_preserve_confidence_accumulation(container_type, package_type):
    containers = objects([[container_type(v) for v in [0, 0, 0.9, 1]]])
    packages = objects([[package_type(v) for v in [0, 0, 1, 1]] for _ in range(7)])
    actual = pp.slot_into_containers(containers, packages)[2]
    expected = pp._slot_into_containers_scalar(containers, packages)[2]
    assert compute_confidence_score(actual) == compute_confidence_score(expected)
    assert [type(score) for score in actual] == [type(score) for score in expected]


def test_mixed_score_types_follow_winning_intersection():
    containers = objects([[0, 0, 0.9, 1], [np.float64(v) for v in [2, 2, 2.9, 3]]])
    packages = objects(
        [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5], [np.float64(v) for v in [4, 4, 5, 5]]]
    )
    actual = pp.slot_into_containers(containers, packages)[2]
    expected = pp._slot_into_containers_scalar(containers, packages)[2]
    assert actual == expected
    assert [type(score) for score in actual] == [type(score) for score in expected]
    assert compute_confidence_score(actual) == compute_confidence_score(expected)
