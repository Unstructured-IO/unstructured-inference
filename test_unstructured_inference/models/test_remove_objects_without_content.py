import copy
import random

import pytest

import unstructured_inference.models.table_postprocess as postprocess


def remove_objects_without_content_by_rect(page_spans, objects):
    """The per-object implementation the vectorized version must match."""
    for obj in objects[:]:
        object_text, _ = postprocess.extract_text_inside_bbox(page_spans, obj["bbox"])
        if len(object_text.strip()) == 0:
            objects.remove(obj)


def random_box(rng, grid):
    if grid:
        # A coarse grid produces zero-area, inverted and exactly-half-overlapping boxes.
        return [rng.randint(0, 6) for _ in range(4)]
    x, y = rng.uniform(-20, 500), rng.uniform(-20, 500)
    return [x, y, x + rng.uniform(-3, 120), y + rng.uniform(-3, 60)]


def random_span(rng, grid):
    span = {
        "bbox": random_box(rng, grid),
        "text": rng.choice(["a", " ", "", "12", "x y"]),
        "span_num": rng.randint(0, 5),
        "line_num": rng.randint(0, 2),
        "block_num": rng.randint(0, 1),
    }
    if rng.random() < 0.4:
        span["flags"] = rng.choice([0, 1, 2, 3])
    return span


@pytest.mark.parametrize("chunk_elements", [1, 2, 5, 1 << 20])
@pytest.mark.parametrize("seed", range(5))
def test_matches_per_object_rect_overlap(monkeypatch, chunk_elements, seed):
    monkeypatch.setattr(postprocess, "_SLOT_CHUNK_ELEMENTS", chunk_elements)
    rng = random.Random(seed)
    for _ in range(100):
        grid = rng.random() < 0.5
        objects = [{"bbox": random_box(rng, grid)} for _ in range(rng.randint(0, 10))]
        if objects and rng.random() < 0.2:
            objects.append(copy.deepcopy(rng.choice(objects)))
        spans = [random_span(rng, grid) for _ in range(rng.randint(0, 30))]
        expected_objects, expected_spans = copy.deepcopy(objects), copy.deepcopy(spans)
        remove_objects_without_content_by_rect(expected_spans, expected_objects)

        postprocess.remove_objects_without_content(spans, objects)

        # Spans are compared too: superscript-flagged spans are marked in place.
        assert (objects, spans) == (expected_objects, expected_spans)


def test_filters_the_callers_list_in_place():
    objects = [{"bbox": [0, 0, 10, 10]}, {"bbox": [50, 50, 60, 60]}]
    spans = [{"bbox": [1, 1, 4, 4], "text": "a", "span_num": 0, "line_num": 0, "block_num": 0}]
    kept = objects[0]

    assert postprocess.remove_objects_without_content(spans, objects) is None
    assert len(objects) == 1
    assert objects[0] is kept


def test_span_needs_half_its_area_inside_and_positive_area():
    objects = [{"bbox": [0, 0, 10, 10]}, {"bbox": [20, 20, 30, 30]}, {"bbox": [40, 40, 50, 50]}]
    spans = [
        # Exactly half inside the first object, which counts.
        {"bbox": [8, 0, 12, 10], "text": "a", "span_num": 0, "line_num": 0, "block_num": 0},
        # Zero area inside the second object, which never counts.
        {"bbox": [25, 25, 25, 28], "text": "b", "span_num": 1, "line_num": 0, "block_num": 0},
        # Less than half inside the third object.
        {"bbox": [48, 40, 53, 50], "text": "c", "span_num": 2, "line_num": 0, "block_num": 0},
    ]

    postprocess.remove_objects_without_content(spans, objects)

    assert objects == [{"bbox": [0, 0, 10, 10]}]


def test_does_not_score_span_pairs_one_at_a_time(monkeypatch):
    def forbid_pairwise_overlap(*args, **kwargs):
        pytest.fail("spans must be scored against all objects at once")

    monkeypatch.setattr(postprocess, "overlaps", forbid_pairwise_overlap)
    objects = [{"bbox": [0, row * 10, 100, row * 10 + 10]} for row in range(50)]
    spans = [
        {
            "bbox": [1, row * 10 + 1, 5, row * 10 + 5],
            "text": "a",
            "span_num": row,
            "line_num": 0,
            "block_num": 0,
        }
        for row in range(0, 50, 2)
    ]

    postprocess.remove_objects_without_content(spans, objects)

    assert objects == [{"bbox": [0, row * 10, 100, row * 10 + 10]} for row in range(0, 50, 2)]
