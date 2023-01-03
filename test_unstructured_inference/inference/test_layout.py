import pytest
from unittest.mock import patch, mock_open

import layoutparser as lp
from layoutparser.elements import Layout, Rectangle, TextBlock
import numpy as np
from PIL import Image

import unstructured_inference.inference.layout as layout
import unstructured_inference.models as models

import unstructured_inference.models.detectron2 as detectron2
import unstructured_inference.models.tesseract as tesseract


@pytest.fixture
def mock_image():
    return Image.new("1", (1, 1))


@pytest.fixture
def mock_page_layout():
    text_rectangle = Rectangle(2, 4, 6, 8)
    text_block = TextBlock(text_rectangle, text="A very repetitive narrative. " * 10, type="Text")

    title_rectangle = Rectangle(1, 2, 3, 4)
    title_block = TextBlock(title_rectangle, text="A Catchy Title", type="Title")

    return Layout([text_block, title_block])


def test_pdf_page_converts_images_to_array(mock_image):
    page = layout.PageLayout(number=0, image=mock_image, layout=Layout())
    assert page.image_array is None

    image_array = page._get_image_array()
    assert isinstance(image_array, np.ndarray)
    assert page.image_array.all() == image_array.all()


def test_ocr(monkeypatch):
    mock_text = "The parrot flies high in the air!"

    class MockOCRAgent:
        def detect(self, *args):
            return mock_text

    monkeypatch.setattr(tesseract, "ocr_agent", MockOCRAgent)
    monkeypatch.setattr(tesseract, "is_pytesseract_available", lambda *args: True)

    image = np.random.randint(12, 24, (40, 40))
    page = layout.PageLayout(number=0, image=image, layout=Layout())
    rectangle = Rectangle(1, 2, 3, 4)
    text_block = TextBlock(rectangle, text=None)

    assert page.ocr(text_block) == mock_text


class MockLayoutModel:
    def __init__(self, layout):
        self.layout = layout

    def detect(self, *args):
        return self.layout


def test_get_page_elements(monkeypatch, mock_page_layout):
    monkeypatch.setattr(detectron2, "load_default_model", lambda: MockLayoutModel(mock_page_layout))
    monkeypatch.setattr(detectron2, "is_detectron2_available", lambda *args: True)

    image = np.random.randint(12, 24, (40, 40))
    page = layout.PageLayout(number=0, image=image, layout=mock_page_layout)

    elements = page.get_elements(inplace=False)

    assert str(elements[0]) == "A Catchy Title"
    assert str(elements[1]).startswith("A very repetitive narrative.")

    page.get_elements(inplace=True)
    assert elements == page.elements


def test_get_page_elements_with_ocr(monkeypatch):
    monkeypatch.setattr(layout.PageLayout, "ocr", lambda *args: "An Even Catchier Title")

    rectangle = Rectangle(2, 4, 6, 8)
    text_block = TextBlock(rectangle, text=None, type="Title")
    doc_layout = Layout([text_block])

    monkeypatch.setattr(detectron2, "load_default_model", lambda: MockLayoutModel(doc_layout))
    monkeypatch.setattr(detectron2, "is_detectron2_available", lambda *args: True)

    image = np.random.randint(12, 24, (40, 40))
    page = layout.PageLayout(number=0, image=image, layout=doc_layout)
    page.get_elements()

    assert str(page) == "An Even Catchier Title"


def test_read_pdf(monkeypatch, mock_page_layout):
    image = np.random.randint(12, 24, (40, 40))
    images = [image, image]

    layouts = Layout([mock_page_layout, mock_page_layout])

    monkeypatch.setattr(detectron2, "load_default_model", lambda: MockLayoutModel(mock_page_layout))
    monkeypatch.setattr(detectron2, "is_detectron2_available", lambda *args: True)

    with patch.object(lp, "load_pdf", return_value=(layouts, images)):
        doc = layout.DocumentLayout.from_file("fake-file.pdf")

        assert str(doc).startswith("A Catchy Title")
        assert str(doc).count("A Catchy Title") == 2  # Once for each page
        assert str(doc).endswith("A very repetitive narrative. ")

        assert doc.pages[0].elements[0].to_dict()["text"] == "A Catchy Title"

        pages = doc.pages
        assert str(doc) == "\n\n".join([str(page) for page in pages])


@pytest.mark.parametrize("model_name", [None, "checkbox", "fake"])
def test_process_data_with_model(monkeypatch, mock_page_layout, model_name):
    monkeypatch.setattr(models, "get_model", lambda x: MockLayoutModel(mock_page_layout))
    monkeypatch.setattr(
        layout.DocumentLayout,
        "from_file",
        lambda *args, **kwargs: layout.DocumentLayout.from_pages([]),
    )
    monkeypatch.setattr(
        models, "load_model", lambda *args, **kwargs: MockLayoutModel(mock_page_layout)
    )
    monkeypatch.setattr(
        models,
        "_get_model_loading_info",
        lambda *args, **kwargs: (
            "fake-binary-path",
            "fake-config-path",
            {0: "Unchecked", 1: "Checked"},
        ),
    )
    with patch("builtins.open", mock_open(read_data=b"000000")):
        assert layout.process_data_with_model(open(""), model_name=model_name)


def test_process_data_with_model_raises_on_invalid_model_name():
    with patch("builtins.open", mock_open(read_data=b"000000")):
        with pytest.raises(models.UnknownModelException):
            layout.process_data_with_model(open(""), model_name="fake")


@pytest.mark.parametrize("model_name", [None, "checkbox"])
def test_process_file_with_model(monkeypatch, mock_page_layout, model_name):
    monkeypatch.setattr(models, "get_model", lambda x: MockLayoutModel(mock_page_layout))
    monkeypatch.setattr(
        layout.DocumentLayout,
        "from_file",
        lambda *args, **kwargs: layout.DocumentLayout.from_pages([]),
    )
    monkeypatch.setattr(
        models, "load_model", lambda *args, **kwargs: MockLayoutModel(mock_page_layout)
    )
    monkeypatch.setattr(
        models,
        "_get_model_loading_info",
        lambda *args, **kwargs: (
            "fake-binary-path",
            "fake-config-path",
            {0: "Unchecked", 1: "Checked"},
        ),
    )
    filename = ""
    assert layout.process_file_with_model(filename, model_name=model_name)


def test_process_file_with_model_raises_on_invalid_model_name():
    with pytest.raises(models.UnknownModelException):
        layout.process_file_with_model("", model_name="fake")
