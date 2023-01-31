import pytest
import os

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
        f"/layout/{filetype}", files={"file": (filename, open(filename, "rb"))}, data=data
    )
    assert response.status_code == expected_response_code


def test_bad_route_404():
    client = TestClient(api.app)
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
    assert response.status_code == 200


def test_layout_v02_local_parsing_image():
    filename = os.path.join("sample-docs", "test-image.jpg")
    from unstructured_inference.models.yolox_model import yolox_local_inference

    # NOTE(benjamin) keep_output = True create a file for each image in
    # localstorage for visualization of the result
    document_layout_1 = yolox_local_inference(filename, type="image", keep_output=True)
    assert len(document_layout_1.pages) == 1
    document_layout_2 = yolox_local_inference(filename, type="image", keep_output=False)
    # NOTE(benjamin) The example image should result in one page result
    assert len(document_layout_2.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 13 detections
    assert len(document_layout_2.pages[0].layout) == 13


def test_layout_v02_local_parsing_pdf():
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    from unstructured_inference.models.yolox_model import yolox_local_inference

    document_layout = yolox_local_inference(filename, type="pdf")
    content = document_layout.tostring()
    assert "Lorem ipsum" in content
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(document_layout.pages[0].layout) == 5


def test_healthcheck(monkeypatch):
    client = TestClient(api.app)
    response = client.get("/healthcheck")
    assert response.status_code == 200
