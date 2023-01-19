import pytest
import os

from fastapi.testclient import TestClient

from unstructured_inference.api import app
from unstructured_inference import models
from unstructured_inference.inference.layout import DocumentLayout
import unstructured_inference.models.detectron2 as detectron2

import jsons


class MockModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pytest.mark.parametrize("filetype, ext", [("pdf", "pdf"), ("image", "png")])
def test_layout_parsing_api(monkeypatch, filetype, ext):
    monkeypatch.setattr(models, "load_model", lambda *args, **kwargs: MockModel(*args, **kwargs))
    monkeypatch.setattr(models, "hf_hub_download", lambda *args, **kwargs: "fake-path")
    monkeypatch.setattr(detectron2, "is_detectron2_available", lambda *args: True)
    monkeypatch.setattr(
        DocumentLayout, "from_file", lambda *args, **kwargs: DocumentLayout.from_pages([])
    )
    monkeypatch.setattr(
        DocumentLayout, "from_image_file", lambda *args, **kwargs: DocumentLayout.from_pages([])
    )

    filename = os.path.join("sample-docs", f"loremipsum.{ext}")

    client = TestClient(app)
    response = client.post(f"/layout/{filetype}", files={"file": (filename, open(filename, "rb"))})
    assert response.status_code == 200

    response = client.post(
        f"/layout/{filetype}",
        files={"file": (filename, open(filename, "rb"))},
        data={"model": "checkbox"},
    )
    assert response.status_code == 200

    response = client.post(
        f"/layout/{filetype}",
        files={"file": (filename, open(filename, "rb"))},
        data={"model": "fake_model"},
    )
    assert response.status_code == 422


def test_bad_route_404():
    client = TestClient(app)
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    response = client.post("/layout/badroute", files={"file": (filename, open(filename, "rb"))})
    assert response.status_code == 404


def test_layout_v02_api_parsing_image():

    filename = os.path.join("sample-docs", "test-image.jpg")

    client = TestClient(app)
    response = client.post(
        "/layout_v1/image",
        headers={"Accept": "multipart/mixed"},
        files=[("files", (filename, open(filename, "rb"), "image/png"))],
    )
    doc_layout = jsons.load(response.json(), DocumentLayout)
    assert len(doc_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 13 detections
    assert len(doc_layout.pages[0]["layout"]) == 13
    # Each detection should have (x1,y1,x2,y2,probability,class) format
    # assert len(response.json()['Detections'][0])==6
    assert response.status_code == 200


def test_layout_v02_api_parsing_pdf():

    filename = os.path.join("sample-docs", "loremipsum.pdf")

    client = TestClient(app)
    response = client.post(
        "/layout_v1/pdf",
        files={"files": (filename, open(filename, "rb"))},
    )
    doc_layout = jsons.load(response.json(), DocumentLayout)
    assert len(doc_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(doc_layout.pages[0]["layout"]) == 5
    # Each detection should have (x1,y1,x2,y2,probability,class) format
    # assert len(response.json()['Detections'][0])==6
    assert response.status_code == 200


def test_layout_v02_local_parsing_image():
    filename = os.path.join("sample-docs", "test-image.jpg")
    from unstructured_inference.layout_model import local_inference

    detections = local_inference(filename, type="image")
    # NOTE(benjamin) The example image should result in one page result
    assert len(detections.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 13 detections
    assert len(detections.pages[0].layout) == 13
    # Each detection should have (x1,y1,x2,y2,probability,class) format
    # assert len(detections['Detections'][0])==6


def test_layout_v02_local_parsing_pdf():
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    from unstructured_inference.layout_model import local_inference

    detections = local_inference(filename, type="pdf")
    assert len(detections.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(detections.pages[0].layout) == 5
    # Each detection should have (x1,y1,x2,y2,probability,class) format
    # assert len(detections['Detections'][0])==6


def test_healthcheck(monkeypatch):
    client = TestClient(app)
    response = client.get("/healthcheck")
    assert response.status_code == 200
