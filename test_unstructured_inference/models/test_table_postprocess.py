"""Exact equivalence checks against selection before the streaming optimization."""

import itertools
import math
import random
from collections import UserList, defaultdict
from copy import deepcopy

import numpy as np
import pytest

from unstructured_inference.models import table_postprocess as postprocess
from unstructured_inference.models.table_postprocess import Rect, sort_objects_by_score


# Frozen baseline from 158db67: keep its arithmetic, sort, and output assembly independent.
def reference_slot_into_containers(
    container_objects,
    package_objects,
    overlap_threshold=0.5,
    forced_assignment=False,
):
    """
    Slot a collection of objects into the container they occupy most (the container which holds the
    largest fraction of the object).
    """
    best_match_scores = []

    container_assignments = [[] for container in container_objects]
    package_assignments = [[] for package in package_objects]

    if len(container_objects) == 0 or len(package_objects) == 0:
        return container_assignments, package_assignments, best_match_scores

    match_scores = defaultdict(dict)
    for package_num, package in enumerate(package_objects):
        match_scores = []
        package_rect = Rect(package["bbox"])
        package_area = package_rect.get_area()
        for container_num, container in enumerate(container_objects):
            container_rect = Rect(container["bbox"])
            intersect_area = container_rect.intersect(Rect(package["bbox"])).get_area()

            if package_area > 0:
                overlap_fraction = intersect_area / package_area

                match_scores.append(
                    {
                        "container": container,
                        "container_num": container_num,
                        "score": overlap_fraction,
                    },
                )

        if len(match_scores) > 0:
            sorted_match_scores = sort_objects_by_score(match_scores)

            best_match_score = sorted_match_scores[0]
            best_match_scores.append(best_match_score["score"])
            if forced_assignment or best_match_score["score"] >= overlap_threshold:
                container_assignments[best_match_score["container_num"]].append(package_num)
                package_assignments[package_num].append(best_match_score["container_num"])

    return container_assignments, package_assignments, best_match_scores


def box(bbox):
    return {"bbox": bbox}


def assert_same(actual, expected):
    assert actual[:2] == expected[:2]
    assert len(actual[2]) == len(expected[2])
    for got, want in zip(actual[2], expected[2]):
        assert type(got) is type(want)
        if want != want:
            assert got != got
        else:
            assert got == want
            if want == 0:
                assert math.copysign(1, got) == math.copysign(1, want)


def check(containers, packages, **kwargs):
    actual = postprocess.slot_into_containers(containers, packages, **kwargs)
    assert_same(actual, reference_slot_into_containers(containers, packages, **kwargs))
    return actual


@pytest.mark.parametrize(("containers", "packages"), [([], []), ([{}], []), ([], [{}])])
def test_empty_inputs(containers, packages):
    assert check(containers, packages) == ([[] for _ in containers], [[] for _ in packages], [])


@pytest.mark.parametrize("forced", [False, True])
def test_mixed_packages_keep_score_cardinality_and_indices(forced):
    containers = [box([0, 0, 4, 4]), box([10, 0, 14, 4])]
    packages = [box(b) for b in ([1, 1, 3, 3], [0, 0, 0, 2], [11, 1, 13, 3], [30, 0, 31, 1])]
    result = check(containers, packages, forced_assignment=forced)
    assert result == (
        [[0, 3], [2]] if forced else [[0], [2]],
        [[0], [], [1], [0] if forced else []],
        [1.0, 1.0, 0.0],
    )


@pytest.mark.parametrize(
    "threshold",
    [
        math.nextafter(0.5, 0),
        0.5,
        math.nextafter(0.5, 1),
        0.001,
        0,
        -1,
        float("inf"),
        -float("inf"),
        float("nan"),
    ],
)
@pytest.mark.parametrize("forced", [False, True])
def test_threshold_and_first_tie(threshold, forced):
    containers = [box([0, 0, 1, 2]), box([0, 0, 1, 2]), box([8, 8, 9, 9])]
    result = check(
        containers, [box([0, 0, 2, 2])], overlap_threshold=threshold, forced_assignment=forced
    )
    assert result[2] == [0.5]
    assert result[1] == ([[0]] if forced or threshold <= 0.5 else [[]])


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 0, 2],
        [0, 0, 2, 0],
        [2, 0, 0, 2],
        [2, 2, 0, 0],
        None,
        [-0.0, -0.0, 1.0, 1.0],
        [0.0, 0.0, 1e-200, 1e-200],
        [0.0, 0.0, float("inf"), 1.0],
        [0.0, 0.0, 1e200, 1e200],
        [float("nan"), 0.0, 2.0, 2.0],
    ],
)
def test_geometry_quirks(bbox):
    ordinary = box([0, 0, 2, 2])
    check(
        [box(bbox), ordinary, box([99, 99, 99, 100])], [box(bbox), ordinary], forced_assignment=True
    )


def test_degenerate_container_still_copies_package():
    assert check([box([99, 99, 99, 100])], [box([0, 0, 1, 1])]) == (
        [[0]],
        [[0]],
        [1.0],
    )


def test_native_nan_sequences_keep_original_first_winner():
    package = box([0.0, 0.0, float("inf"), 1.0])
    small, full = box([0.0, 0.0, 1.0, 1.0]), package
    patterns = itertools.chain.from_iterable(
        itertools.product((small, full), repeat=n) for n in range(1, 9)
    )
    for containers in patterns:
        check(containers, [package], forced_assignment=True)
    for count in (31, 32, 63, 64, 65, 127, 128, 129, 257):
        for pair in ((small, full), (full, small)):
            check(list(pair) * count, [package], forced_assignment=True)


@pytest.mark.parametrize("container_collection", [list, tuple, UserList])
def test_generated_finite_geometry(container_collection):
    rng = random.Random(20261015)
    for _ in range(40):

        def generated_box():
            # Include degenerate and inverted dimensions, both signs, and repeated coordinates.
            x, y = rng.randint(-20, 20), rng.randint(-20, 20)
            return box([x, y, x + rng.randint(-4, 12), y + rng.randint(-4, 12)])

        containers = container_collection(generated_box() for _ in range(rng.randrange(1, 18)))
        packages = [generated_box() for _ in range(rng.randrange(1, 22))]
        saved = deepcopy((containers, packages))
        for threshold, force in ((0.5, False), (0.001, False), (0, False), (1.1, True)):
            check(containers, packages, overlap_threshold=threshold, forced_assignment=force)
        assert (containers, packages) == saved


def test_numpy_compatibility_including_mixed_nan_scores():
    package = box([0.0, 0.0, 1e20, 1e20])
    containers = [
        box([0.0, 0.0, 0.25e20, 1e20]),
        box(list(np.array(package["bbox"], dtype=np.float32))),
        box([0.0, 0.0, 0.75e20, 1e20]),
    ]
    with np.errstate(all="ignore"):
        check(containers, [package])
        check(list(reversed(containers)), [package], forced_assignment=True)
        for dtype in (np.float32, np.float64, np.int64):
            check(
                [box(np.array([0, 0, 4, 4], dtype=dtype))],
                [box(np.array([1, 1, 3, 3], dtype=dtype))],
            )
        # Mixed precisions can overflow the intersection while the denominator remains finite.
        check(
            [box(np.array([0, 0, 5e19, 5e19], dtype=np.float32))],
            [box(np.array([0, 0, 1e20, 1e20], dtype=np.float64))],
        )


@pytest.mark.parametrize("package", [box([0, 0, 1, 1]), box([0, 0, 0, 1])])
@pytest.mark.parametrize("bad_container", [{}, box([1]), box(1)])
def test_do_not_skip_later_invalid_geometry(package, bad_container):
    containers = [box([0, 0, 1, 1]), bad_container]
    with pytest.raises((KeyError, IndexError, TypeError)) as error:
        reference_slot_into_containers(containers, [package])
    with pytest.raises(type(error.value)):
        postprocess.slot_into_containers(containers, [package])


def test_force_short_circuits_threshold_and_results_are_independent():
    class InvalidThreshold:
        def __le__(self, other):
            raise AssertionError("forced assignment must not compare threshold")

    containers = [box([0, 0, 1, 1])] * 2
    packages = [box([0, 0, 1, 1])] * 2
    result = check(
        containers, packages, forced_assignment=True, overlap_threshold=InvalidThreshold()
    )
    assert result == ([[0, 1], []], [[0], [0]], [1.0, 1.0])
    result[0][0].append(999)
    assert check(containers, packages)[0] == [[0, 1], []]


def test_nms_ties_follow_detection_score_order():
    containers = [{"bbox": [0, 0, 4, 4], "score": 0.5}, {"bbox": [0, 0, 4, 4], "score": 0.9}]
    assert postprocess.nms_by_containment(containers, [box([1, 1, 2, 2])]) == [containers[1]]


@pytest.mark.parametrize("with_spanning_cell", [False, True])
def test_structure_to_cells_complete_parity(monkeypatch, with_spanning_cell):
    from unstructured_inference.models.tables import structure_to_cells

    structure = {
        "rows": [{"bbox": [0, 0, 20, 10]}, {"bbox": [0, 10, 20, 20]}],
        "columns": [{"bbox": [0, 0, 10, 20]}, {"bbox": [10, 0, 20, 20]}],
        "spanning cells": (
            [{"bbox": [0, 0, 20, 10], "score": 0.9, "projected row header": False}]
            if with_spanning_cell
            else []
        ),
    }
    tokens = [
        {"text": text, "bbox": bbox, "span_num": n, "line_num": n, "block_num": 0}
        for n, (text, bbox) in enumerate(
            [
                ("-12.50%", [1, 1, 9, 9]),
                ("Header", [11, 1, 19, 9]),
                ("0", [1, 11, 9, 19]),
                ("23.01", [11, 11, 19, 19]),
                ("empty", [5, 5, 5, 8]),
                ("unmatched", [30, 30, 31, 31]),
            ]
        )
    ]
    candidate = structure_to_cells(deepcopy(structure), deepcopy(tokens))
    with monkeypatch.context() as patch:
        patch.setattr(postprocess, "slot_into_containers", reference_slot_into_containers)
        baseline = structure_to_cells(deepcopy(structure), deepcopy(tokens))
    assert candidate == baseline
