import os
import shutil

import jsons
import pytest
from fastapi.testclient import TestClient

from unstructured_inference import api
from unstructured_inference.inference.layout import DocumentLayout
from unstructured_inference.models.yolox import yolox_local_inference  # DocumentLayout #maybe


@pytest.mark.skipif(os.environ.get("TEST_LONG") or not os.environ.get("TEST_LONG") == "1", reason="Not need to run long test")
def test_layout_v02_api_parsing_image():
    filename = os.path.join("sample-docs", "test-image.jpg")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/image",
        headers={"Accept": "multipart/mixed"},
        files=[("file", (filename, open(filename, "rb"), "image/png"))],
        data={"version": "yolox"},
    )
    doc_layout = jsons.load(response.json(), DocumentLayout)
    assert len(doc_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 13 detections
    assert len(doc_layout.pages[0]["layout"]) == 13
    assert response.status_code == 200


@pytest.mark.skipif(os.environ.get("TEST_LONG") or not os.environ.get("TEST_LONG") == "1", reason="Not need to run long test")
def test_layout_v02_api_parsing_pdf():
    filename = os.path.join("sample-docs", "loremipsum.pdf")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/pdf",
        files={"file": (filename, open(filename, "rb"))},
        data={"version": "yolox"},
    )
    doc_layout = jsons.load(response.json(), DocumentLayout)
    assert len(doc_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(doc_layout.pages[0]["layout"]) == 5
    assert response.status_code == 200


@pytest.mark.skipif(os.environ.get("TEST_LONG") or not os.environ.get("TEST_LONG") == "1", reason="Not need to run long test")
def test_layout_v02_api_parsing_pdf_ocr():
    filename = os.path.join("sample-docs", "non-embedded.pdf")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/pdf",
        files={"file": (filename, open(filename, "rb"))},
        data={"force_ocr": True, "version": "yolox"},
    )
    doc_layout = jsons.load(response.json(), DocumentLayout)
    assert len(doc_layout.pages) == 10
    assert len(doc_layout.pages[0]["layout"]) > 1
    assert response.status_code == 200


@pytest.mark.skipif(os.environ.get("TEST_LONG") or not os.environ.get("TEST_LONG") == "1", reason="Not need to run long test")
def test_layout_v02_local_parsing_image():
    filename = os.path.join("sample-docs", "test-image.jpg")
    OUTPUT_DIR = "yolox_output"
    # NOTE(benjamin) keep_output = True create a file for each image in
    # localstorage for visualization of the result
    if os.path.exists(OUTPUT_DIR):
        # NOTE(benjamin): should delete the default output folder on test?
        shutil.rmtree(OUTPUT_DIR)
    document_layout_1 = yolox_local_inference(
        filename, type="image", output_directory=OUTPUT_DIR, version="yolox"
    )
    assert len(document_layout_1.pages) == 1
    document_layout_2 = yolox_local_inference(filename, type="image", version="yolox")
    # NOTE(benjamin) The example image should result in one page result
    assert len(document_layout_2.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 13 detections
    assert len(document_layout_2.pages[0].layout) == 13


@pytest.mark.skipif(os.environ.get("TEST_LONG") or not os.environ.get("TEST_LONG") == "1", reason="Not need to run long test")
def test_layout_v02_local_parsing_pdf():
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    document_layout = yolox_local_inference(filename, type="pdf", version="yolox")
    content = document_layout.to_string()
    assert "Lorem ipsum" in content
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(document_layout.pages[0].layout) == 5


@pytest.mark.skipif(os.environ.get("TEST_LONG") or not os.environ.get("TEST_LONG") == "1", reason="Not need to run long test")
def test_layout_v02_local_parsing_empty_pdf():
    filename = os.path.join("sample-docs", "empty-document.pdf")
    document_layout = yolox_local_inference(filename, type="pdf", version="yolox")
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(document_layout.pages[0].layout) == 0


# Only short test below


def test_layout_v02_api_parsing_image_soft():
    filename = os.path.join("sample-docs", "test-image.jpg")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/image",
        headers={"Accept": "multipart/mixed"},
        files=[("file", (filename, open(filename, "rb"), "image/png"))],
        data={"version": "yolox_tiny"},
    )
    doc_layout = jsons.load(response.json(), DocumentLayout)
    assert len(doc_layout.pages) == 1
    # NOTE(benjamin) Soft version of the test, run make test-long in order to run with full model
    assert len(doc_layout.pages[0]["layout"]) > 0
    assert response.status_code == 200


def test_layout_v02_api_parsing_pdf_soft():
    filename = os.path.join("sample-docs", "loremipsum.pdf")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/pdf",
        files={"file": (filename, open(filename, "rb"))},
        data={"version": "yolox_tiny"},
    )
    doc_layout = jsons.load(response.json(), DocumentLayout)
    assert len(doc_layout.pages) == 1
    # NOTE(benjamin) Soft version of the test, run make test-long in order to run with full model
    assert len(doc_layout.pages[0]["layout"]) > 0
    assert response.status_code == 200


def test_layout_v02_api_parsing_pdf_ocr_soft():
    filename = os.path.join("sample-docs", "non-embedded.pdf")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/pdf",
        files={"file": (filename, open(filename, "rb"))},
        data={"force_ocr": True, "version": "yolox_tiny"},
    )
    doc_layout = jsons.load(response.json(), DocumentLayout)
    assert len(doc_layout.pages) == 10
    assert len(doc_layout.pages[0]["layout"]) > 1
    assert response.status_code == 200


def test_layout_v02_local_parsing_image_soft():
    filename = os.path.join("sample-docs", "test-image.jpg")
    OUTPUT_DIR = "yolox_output"
    # NOTE(benjamin) keep_output = True create a file for each image in
    # localstorage for visualization of the result
    if os.path.exists(OUTPUT_DIR):
        # NOTE(benjamin): should delete the default output folder on test?
        shutil.rmtree(OUTPUT_DIR)
    document_layout_1 = yolox_local_inference(
        filename, type="image", output_directory=OUTPUT_DIR, version="yolox_tiny"
    )
    assert len(document_layout_1.pages) == 1
    document_layout_2 = yolox_local_inference(filename, type="image", version="yolox_tiny")
    # NOTE(benjamin) The example image should result in one page result
    assert len(document_layout_2.pages) == 1
    # NOTE(benjamin) Soft version of the test, run make test-long in order to run with full model
    assert len(document_layout_2.pages[0].layout) > 0


def test_layout_v02_local_parsing_pdf_soft():
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    document_layout = yolox_local_inference(filename, type="pdf", version="yolox_tiny")
    content = document_layout.to_string()
    assert "Lorem ipsum" in content
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) Soft version of the test, run make test-long in order to run with full model
    assert len(document_layout.pages[0].layout) > 0


def test_layout_v02_local_parsing_empty_pdf_soft():
    filename = os.path.join("sample-docs", "empty-document.pdf")
    document_layout = yolox_local_inference(filename, type="pdf", version="yolox_tiny")
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(document_layout.pages[0].layout) == 0
