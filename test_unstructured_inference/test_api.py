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

def test_healthcheck(monkeypatch):
    client = TestClient(api.app)
    response = client.get("/healthcheck")
    assert response.status_code == 200
