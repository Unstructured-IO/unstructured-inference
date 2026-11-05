"""Synthetic geometry-only benchmark; no document data or model weights required.

Run from the repository root: python -m benchmarks.table_geometry
Reports interleaved median times after checking output parity. These timings
exclude OCR, neural inference, and HTML generation.
"""

import json
import platform
from statistics import median
from time import perf_counter

import numpy as np

from unstructured_inference.models import table_postprocess as pp


def make_grid(cell_count):
    cells = []
    tokens = []
    for index in range(cell_count):
        x, y = (index % 10) * 100, (index // 10) * 40
        cells.append({"bbox": [x, y, x + 100, y + 40]})
        for word in range(4):
            left = x + 2 + word * 24
            tokens.append({"bbox": [left, y + 5, left + 20, y + 15]})
    return cells, tokens


def run_geometry(cells, tokens, scalar):
    assign = pp._slot_into_containers_scalar if scalar else pp.slot_into_containers
    subset = pp._get_bbox_span_subset_scalar if scalar else pp.get_bbox_span_subset
    return assign(cells, tokens), [subset(tokens, cell["bbox"]) for cell in cells]


def main():
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "system": platform.system(),
                "machine": platform.machine(),
                "repeats": 7,
            },
        ),
    )
    for cell_count in (10, 100, 500):
        cells, tokens = make_grid(cell_count)
        assert run_geometry(cells, tokens, True) == run_geometry(cells, tokens, False)
        samples = {True: [], False: []}
        for repeat in range(7):
            for scalar in (True, False) if repeat % 2 == 0 else (False, True):
                start = perf_counter()
                run_geometry(cells, tokens, scalar)
                samples[scalar].append((perf_counter() - start) * 1000)
        baseline, candidate = median(samples[True]), median(samples[False])
        print(
            json.dumps(
                {
                    "cells": cell_count,
                    "tokens": len(tokens),
                    "scalar_ms": round(baseline, 3),
                    "vectorized_ms": round(candidate, 3),
                    "speedup": round(baseline / candidate, 2),
                    "exact_output_match": True,
                },
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
