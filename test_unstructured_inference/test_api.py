import os
import shutil

import jsons
import pytest
from fastapi.testclient import TestClient

from unstructured_inference import api
from unstructured_inference.models import base as models
from unstructured_inference.inference.layout import DocumentLayout
from unstructured_inference.models.yolox import yolox_local_inference  # DocumentLayout #maybe


class MockModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def get_doc_layout(*args, **kwargs):
    return DocumentLayout.from_pages([])


def raise_unknown_model_exception(*args, **kwargs):
    raise models.UnknownModelException()


@pytest.mark.parametrize(
    "filetype, ext, data, process_func, expected_response_code",
    [
        ("pdf", "pdf", None, get_doc_layout, 200),
        ("pdf", "pdf", {"model": "checkbox"}, get_doc_layout, 200),
        ("pdf", "pdf", {"model": "fake_model"}, raise_unknown_model_exception, 422),
        ("image", "png", None, get_doc_layout, 200),
        ("image", "png", {"model": "checkbox"}, get_doc_layout, 200),
        ("image", "png", {"model": "fake_model"}, raise_unknown_model_exception, 422),
    ],
)
def test_layout_parsing_api(monkeypatch, filetype, ext, data, process_func, expected_response_code):
    monkeypatch.setattr(api, "process_data_with_model", process_func)

    filename = os.path.join("sample-docs", f"loremipsum.{ext}")

    client = TestClient(api.app)
    response = client.post(
        f"/layout/detectron/{filetype}",
        files={"file": (filename, open(filename, "rb"))},
        data=data,
    )
    assert response.status_code == expected_response_code


def test_bad_route_404():
    client = TestClient(api.app)
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    response = client.post("/layout/badroute", files={"file": (filename, open(filename, "rb"))})
    assert response.status_code == 404


def test_layout_v02_api_parsing_image():

    filename = os.path.join("sample-docs", "test-image.jpg")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/image",
        headers={"Accept": "multipart/mixed"},
        files=[("file", (filename, open(filename, "rb"), "image/png"))],
    )
    doc_layout = jsons.load(response.json(), DocumentLayout)
    assert len(doc_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 13 detections
    assert len(doc_layout.pages[0]["layout"]) == 13
    assert response.status_code == 200


def test_layout_v02_api_parsing_pdf():

    filename = os.path.join("sample-docs", "loremipsum.pdf")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/pdf",
        files={"file": (filename, open(filename, "rb"))},
    )
    doc_layout = jsons.load(response.json(), DocumentLayout)
    assert len(doc_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(doc_layout.pages[0]["layout"]) == 5
    assert response.status_code == 200


def test_layout_v02_api_parsing_pdf_ocr():

    filename = os.path.join("sample-docs", "non-embedded.pdf")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/pdf",
        files={"file": (filename, open(filename, "rb"))},
        data={"force_ocr": True},
    )
    doc_layout = jsons.load(response.json(), DocumentLayout)
    assert len(doc_layout.pages) == 10
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(doc_layout.pages[0]["layout"]) > 1
    assert response.status_code == 200


def test_layout_v02_local_parsing_image():
    filename = os.path.join("sample-docs", "test-image.jpg")
    OUTPUT_DIR = "yolox_output"
    # NOTE(benjamin) keep_output = True create a file for each image in
    # localstorage for visualization of the result
    if os.path.exists(OUTPUT_DIR):
        # NOTE(benjamin): should delete the default output folder on test?
        shutil.rmtree(OUTPUT_DIR)
    document_layout_1 = yolox_local_inference(filename, type="image", output_directory=OUTPUT_DIR)
    assert len(document_layout_1.pages) == 1
    document_layout_2 = yolox_local_inference(filename, type="image")
    # NOTE(benjamin) The example image should result in one page result
    assert len(document_layout_2.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 13 detections
    assert len(document_layout_2.pages[0].layout) == 13


def test_layout_v02_local_parsing_pdf():
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    document_layout = yolox_local_inference(filename, type="pdf")
    content = document_layout.to_string()
    assert "Lorem ipsum" in content
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(document_layout.pages[0].layout) == 5


def test_layout_v02_local_parsing_empty_pdf():
    filename = os.path.join("sample-docs", "empty-document.pdf")
    document_layout = yolox_local_inference(filename, type="pdf")
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(document_layout.pages[0].layout) == 0


def test_healthcheck(monkeypatch):
    client = TestClient(api.app)
    response = client.get("/healthcheck")
    assert response.status_code == 200
