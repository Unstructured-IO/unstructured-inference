import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "unstructured_inference.inference.layout",
        "unstructured_inference.models.base",
        "unstructured_inference.models.detectron2onnx",
        "unstructured_inference.visualize",
    ],
)
def test_import_does_not_load_onnx_or_matplotlib(module):
    # A fresh interpreter, since this test session may already have loaded either library.
    code = (
        f"import sys, {module}\n"
        "loaded = sorted({name.split('.')[0] for name in sys.modules} & {'onnx', 'matplotlib'})\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
