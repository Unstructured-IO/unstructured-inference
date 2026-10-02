"""Measure layout deduplication on a large connected group or disconnected regions.

Run the same script with baseline and candidate packages under the same memory limit:
    PYTHONPATH=. python scripts/performance/benchmark_layout_cleanup_memory.py \
        --count 8000 --case connected --pipeline model
A connected chain reaches one large cleanup group while its graph has only nearby edges.
"""

import argparse
import hashlib
import json
import resource
import sys
import time

import numpy as np
from PIL import Image

from unstructured_inference.inference.layout import PageLayout
from unstructured_inference.inference.layoutelement import LayoutElements, clean_layoutelements
from unstructured_inference.models.unstructuredmodel import UnstructuredObjectDetectionModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8000)
    parser.add_argument(
        "--case", choices=["connected", "sparse", "duplicates"], default="connected"
    )
    parser.add_argument("--pipeline", choices=["cleanup", "model"], default="model")
    args = parser.parse_args()
    if args.count < 2:
        parser.error("--count must be at least 2")

    x = np.arange(args.count, dtype=float)
    if args.case == "sparse":
        x *= 20
    elif args.case == "duplicates":
        x[:] = 0
    coords = np.column_stack((x, np.zeros(args.count), x + 1, np.ones(args.count)))
    regions = LayoutElements(
        element_coords=coords,
        texts=np.arange(args.count).astype(str),
        element_probs=np.full(args.count, 0.9),
        element_class_ids=np.zeros(args.count, dtype=int),
        element_class_id_map={0: "Text"},
    )
    regions.routing = "text"
    regions.routing_score = 0.9

    class Detector(UnstructuredObjectDetectionModel):
        def initialize(self):
            pass

        def predict(self, image):
            return regions

    print(
        json.dumps(
            {"phase": "start", "count": args.count, "case": args.case, "pipeline": args.pipeline}
        ),
        flush=True,
    )
    start = time.perf_counter()
    if args.pipeline == "model":
        page = PageLayout.from_image(Image.new("RGB", (32, 32)), detection_model=Detector())
        output = page.elements_array
    else:
        output = clean_layoutelements(regions)
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024**2 if sys.platform == "darwin" else 1024)
    payload = {
        attr: getattr(output, attr).tolist()
        for attr in ("element_coords", *output._optional_array_attributes)
    }
    payload["element_class_id_map"] = output.element_class_id_map
    payload["routing"] = output.routing
    payload["routing_score"] = output.routing_score
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    print(
        json.dumps(
            {
                "phase": "complete",
                "count": args.count,
                "case": args.case,
                "pipeline": args.pipeline,
                "output_count": len(output),
                "output_sha256": digest,
                "seconds": round(elapsed, 6),
                "peak_rss_mib": round(peak_mib, 2),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
