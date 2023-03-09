import os

import pytest
from fastapi.testclient import TestClient

from unstructured_inference import api
from unstructured_inference.models import base as models
from unstructured_inference.inference.layout import DocumentLayout


class MockModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def get_doc_layout(*args, **kwargs):
    return DocumentLayout.from_pages([])


def raise_unknown_model_exception(*args, **kwargs):
    raise models.UnknownModelException()


@pytest.mark.parametrize(
    "filetype, ext, modeltype, process_func, expected_response_code",
    [
        ("pdf", "pdf", "default", get_doc_layout, 200),
        ("pdf", "pdf", "checkbox", get_doc_layout, 200),
        ("pdf", "pdf", "fake_model", raise_unknown_model_exception, 422),
        ("image", "png", "default", get_doc_layout, 200),
        ("image", "png", "checkbox", get_doc_layout, 200),
        ("image", "png", "fake_model", raise_unknown_model_exception, 422),
    ],
)
def test_layout_parsing_api(
    monkeypatch, filetype, ext, modeltype, process_func, expected_response_code
):
    monkeypatch.setattr(api, "process_data_with_model", process_func)

    filename = os.path.join("sample-docs", f"loremipsum.{ext}")

    client = TestClient(api.app)
    response = client.post(
        f"/layout/{modeltype}/{filetype}",
        files={"file": (filename, open(filename, "rb"))},
    )
    assert response.status_code == expected_response_code


def test_bad_route_404():
    client = TestClient(api.app)
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    response = client.post(
        "/layout/detectron/badroute", files={"file": (filename, open(filename, "rb"))}
    )
    assert response.status_code == 404


@pytest.mark.slow
def test_layout_yolox_api_parsing_image():
    filename = os.path.join("sample-docs", "test-image.jpg")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/image",
        headers={"Accept": "multipart/mixed"},
        files=[("file", (filename, open(filename, "rb"), "image/png"))],
    )
    doc_layout = response.json()
    assert len(doc_layout["pages"]) == 1
    # NOTE(benjamin) The example sent to the test contains 13 detections
    assert len(doc_layout["pages"][0]["elements"]) == 13
    assert response.status_code == 200


@pytest.mark.slow
def test_layout_yolox_api_parsing_pdf():
    filename = os.path.join("sample-docs", "loremipsum.pdf")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/pdf",
        files={"file": (filename, open(filename, "rb"))},
    )
    doc_layout = response.json()
    assert len(doc_layout["pages"]) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(doc_layout["pages"][0]["elements"]) == 5
    assert response.status_code == 200


@pytest.mark.slow
def test_layout_yolox_api_parsing_pdf_ocr():
    filename = os.path.join("sample-docs", "non-embedded.pdf")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox/pdf",
        files={"file": (filename, open(filename, "rb"))},
        data={"force_ocr": True},
    )
    doc_layout = response.json()
    assert len(doc_layout["pages"]) == 10
    assert len(doc_layout["pages"][0]["elements"]) > 1
    assert response.status_code == 200


def test_healthcheck(monkeypatch):
    client = TestClient(api.app)
    response = client.get("/healthcheck")
    assert response.status_code == 200


def test_layout_yolox_api_parsing_image_soft():
    filename = os.path.join("sample-docs", "test-image.jpg")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox_tiny/image",
        headers={"Accept": "multipart/mixed"},
        files=[("file", (filename, open(filename, "rb"), "image/png"))],
    )
    doc_layout = response.json()
    assert len(doc_layout["pages"]) == 1
    # NOTE(benjamin) Soft version of the test, run make test-long in order to run with full model
    assert len(doc_layout["pages"][0]["elements"]) > 0
    assert response.status_code == 200

def test_layout_yolox_api_parsing_image_RGBA_soft():
    filename = os.path.join("sample-docs", "RGBA_image.png")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox_tiny/image",
        headers={"Accept": "multipart/mixed"},
        files=[("file", (filename, open(filename, "rb"), "image/png"))],
    )
    doc_layout = response.json()
    assert len(doc_layout["pages"]) == 1
    # NOTE(benjamin) Soft version of the test, run make test-long in order to run with full model
    assert len(doc_layout["pages"][0]["elements"]) > 0
    assert response.status_code == 200


def test_layout_yolox_api_parsing_pdf_soft():
    filename = os.path.join("sample-docs", "loremipsum.pdf")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox_tiny/pdf",
        files={"file": (filename, open(filename, "rb"))},
    )
    doc_layout = response.json()
    assert len(doc_layout["pages"]) == 1
    # NOTE(benjamin) Soft version of the test, run make test-long in order to run with full model
    assert len(doc_layout["pages"][0]["elements"]) > 0
    assert response.status_code == 200


def test_layout_yolox_api_parsing_pdf_ocr_soft():
    filename = os.path.join("sample-docs", "non-embedded.pdf")

    client = TestClient(api.app)
    response = client.post(
        "/layout/yolox_tiny/pdf",
        files={"file": (filename, open(filename, "rb"))},
        data={"force_ocr": True},
    )
    doc_layout = response.json()
    assert len(doc_layout["pages"]) == 10
    assert len(doc_layout["pages"][0]["elements"]) > 1
    assert response.status_code == 200
