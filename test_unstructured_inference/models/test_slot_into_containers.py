import random

import pytest

import unstructured_inference.models.table_postprocess as postprocess
from unstructured_inference.models.table_postprocess import Rect, slot_into_containers


def slot_into_containers_by_rect(containers, packages, overlap_threshold, forced_assignment):
    """The pairwise Rect implementation the vectorized version must match."""
    container_assignments = [[] for _ in containers]
    package_assignments = [[] for _ in packages]
    best_match_scores = []
    if not containers or not packages:
        return container_assignments, package_assignments, best_match_scores
    for package_num, package in enumerate(packages):
        package_area = Rect(package["bbox"]).get_area()
        if not package_area > 0:
            continue
        scores = [
            Rect(container["bbox"]).intersect(Rect(package["bbox"])).get_area() / package_area
            for container in containers
        ]
        container_num = scores.index(max(scores))
        best_match_scores.append(scores[container_num])
        if forced_assignment or scores[container_num] >= overlap_threshold:
            container_assignments[container_num].append(package_num)
            package_assignments[package_num].append(container_num)
    return container_assignments, package_assignments, best_match_scores


def random_box(rng, grid):
    if grid:
        # A coarse grid produces exact score ties and zero-area or inverted boxes.
        return [rng.randint(0, 6) for _ in range(4)]
    x, y = rng.uniform(-50, 1000), rng.uniform(-50, 1000)
    return [x, y, x + rng.uniform(-5, 80), y + rng.uniform(-5, 40)]


@pytest.mark.parametrize("chunk_elements", [1, 3, 7, 1 << 20])
@pytest.mark.parametrize("seed", range(5))
def test_matches_pairwise_rect_scores(monkeypatch, chunk_elements, seed):
    monkeypatch.setattr(postprocess, "_SLOT_CHUNK_ELEMENTS", chunk_elements)
    rng = random.Random(seed)
    for _ in range(100):
        grid = rng.random() < 0.5
        containers = [{"bbox": random_box(rng, grid)} for _ in range(rng.randint(0, 12))]
        packages = [{"bbox": random_box(rng, grid)} for _ in range(rng.randint(0, 40))]
        threshold = rng.choice([0.0, 0.001, 0.5, 1.0])
        forced = rng.random() < 0.3
        result = slot_into_containers(containers, packages, threshold, forced)
        assert result == slot_into_containers_by_rect(containers, packages, threshold, forced)
        assert all(type(score) is float for score in result[2])


def test_zero_area_container_holds_every_package():
    containers = [{"bbox": [0, 0, 10, 10]}, {"bbox": [50, 50, 50, 60]}]
    packages = [{"bbox": [1, 1, 2, 2]}, {"bbox": [100, 100, 101, 101]}]
    # The first package ties with the real container, which comes first; the second
    # lies outside it and lands in the zero-area container.
    assert slot_into_containers(containers, packages) == ([[0], [1]], [[0], [1]], [1.0, 1.0])


def test_zero_area_package_is_not_scored():
    containers = [{"bbox": [0, 0, 10, 10]}]
    packages = [{"bbox": [1, 1, 1, 5]}, {"bbox": [1, 1, 3, 3]}]
    assert slot_into_containers(containers, packages) == ([[1]], [[], [0]], [1.0])


def test_tied_scores_choose_first_container():
    containers = [{"bbox": [0, 0, 10, 10]}, {"bbox": [0, 0, 10, 10]}]
    packages = [{"bbox": [2, 2, 4, 4]}]
    assert slot_into_containers(containers, packages) == ([[0], []], [[0]], [1.0])
