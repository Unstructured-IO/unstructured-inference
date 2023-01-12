import pytest
import os

from fastapi.testclient import TestClient

from unstructured_inference.api import app
from unstructured_inference import models
from unstructured_inference.inference.layout import DocumentLayout
import unstructured_inference.models.detectron2 as detectron2


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


def test_healthcheck(monkeypatch):
    client = TestClient(app)
    response = client.get("/healthcheck")
    assert response.status_code == 200
