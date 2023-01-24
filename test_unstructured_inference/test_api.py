import pytest
import os

from fastapi.testclient import TestClient

from unstructured_inference.api import app
from unstructured_inference.models import base as models
from unstructured_inference.inference.layout import DocumentLayout
import unstructured_inference.models.detectron2 as detectron2


class MockModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "filetype, ext, data, response_code",
    [
        ("pdf", "pdf", None, 200),
        ("pdf", "pdf", {"model": "checkbox"}, 200),
        ("pdf", "pdf", {"model": "fake_model"}, 422),
        ("image", "png", None, 200),
        ("image", "png", {"model": "checkbox"}, 200),
        ("image", "png", {"model": "fake_model"}, 422),
    ],
)
def test_layout_parsing_api(monkeypatch, filetype, ext, data, response_code):
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
    response = client.post(
        f"/layout/{filetype}", files={"file": (filename, open(filename, "rb"))}, data=data
    )
    assert response.status_code == response_code


def test_bad_route_404():
    client = TestClient(app)
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    response = client.post("/layout/badroute", files={"file": (filename, open(filename, "rb"))})
    assert response.status_code == 404


def test_healthcheck(monkeypatch):
    client = TestClient(app)
    response = client.get("/healthcheck")
    assert response.status_code == 200
