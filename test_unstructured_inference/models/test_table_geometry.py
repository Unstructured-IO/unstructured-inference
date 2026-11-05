"""Output equivalence, stable assignment and unusual rectangle semantics."""

import json
import random

import numpy as np
import pytest

from unstructured_inference.models import table_postprocess as pp


def objects(boxes):
    return [{"bbox": box} for box in boxes]


@pytest.mark.parametrize("threshold", [0, 0.5, np.nextafter(0.5, 1), 1, 1.1])
@pytest.mark.parametrize("forced", [False, True])
def test_geometry_matches_scalar_with_chunking(monkeypatch, threshold, forced):
    rng = random.Random(20261105)

    def random_boxes(count):
        result = []
        for _ in range(count):
            x, y = rng.uniform(-500, 500), rng.uniform(-500, 500)
            result.append([x, y, x + rng.uniform(0.01, 500), y + rng.uniform(0.01, 500)])
        return objects(result)

    containers = random_boxes(11)
    containers.extend(containers[:2])  # Equal-score containers retain their original order.
    packages = random_boxes(31)
    monkeypatch.setattr(pp, "_MAX_GEOMETRY_PAIRS", 39)  # Exercise many package chunks.
    assert pp.slot_into_containers(containers, packages, threshold, forced) == (
        pp._slot_into_containers_scalar(containers, packages, threshold, forced)
    )
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
