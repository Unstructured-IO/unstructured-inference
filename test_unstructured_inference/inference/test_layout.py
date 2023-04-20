from functools import partial
from itertools import product
import pytest
import tempfile
from unittest.mock import patch, mock_open

import numpy as np
from PIL import Image

import unstructured_inference.inference.layout as layout
import unstructured_inference.inference.elements as elements
import unstructured_inference.models.base as models
import unstructured_inference.models.detectron2 as detectron2
import unstructured_inference.models.tesseract as tesseract


@pytest.fixture
def mock_image():
    return Image.new("1", (1, 1))


@pytest.fixture
def mock_page_layout():
    text_block = layout.EmbeddedTextRegion(2, 4, 6, 8, text="A very repetitive narrative. " * 10)

    title_block = layout.EmbeddedTextRegion(1, 2, 3, 4, text="A Catchy Title")

    return [text_block, title_block]


def test_pdf_page_converts_images_to_array(mock_image):
    page = layout.PageLayout(number=0, image=mock_image, layout=[])
    assert page.image_array is None

    image_array = page._get_image_array()
    assert isinstance(image_array, np.ndarray)
    assert page.image_array.all() == image_array.all()


def test_ocr(monkeypatch):
    mock_text = "The parrot flies high in the air!"

    class MockOCRAgent:
        def detect(self, *args):
            return mock_text

    monkeypatch.setattr(tesseract, "ocr_agents", {"eng": MockOCRAgent})
    monkeypatch.setattr(tesseract, "is_pytesseract_available", lambda *args: True)

    image = Image.fromarray(np.random.randint(12, 24, (40, 40)), mode="RGB")
    text_block = layout.TextRegion(1, 2, 3, 4, text=None)

    assert elements.ocr(text_block, image=image) == mock_text


class MockLayoutModel:
    def __init__(self, layout):
        self.layout_return = layout

    def __call__(self, *args):
        return self.layout_return

    def initialize(self, *args, **kwargs):
        pass


def test_get_page_elements(monkeypatch, mock_page_layout):
    image = np.random.randint(12, 24, (40, 40))
    page = layout.PageLayout(
        number=0, image=image, layout=mock_page_layout, model=MockLayoutModel(mock_page_layout)
    )

    elements = page.get_elements_with_model(inplace=False)

    assert str(elements[0]) == "A Catchy Title"
    assert str(elements[1]).startswith("A very repetitive narrative.")

    page.get_elements_with_model(inplace=True)
    assert elements == page.elements


class MockPool:
    def map(self, f, xs):
        return [f(x) for x in xs]

    def close(self):
        pass

    def join(self):
        pass


def test_get_page_elements_with_ocr(monkeypatch):
    text_block = layout.TextRegion(2, 4, 6, 8, text=None)
    image_block = layout.ImageTextRegion(8, 14, 16, 18)
    doc_layout = [text_block, image_block]

    monkeypatch.setattr(detectron2, "is_detectron2_available", lambda *args: True)
    monkeypatch.setattr(elements, "ocr", lambda *args, **kwargs: "An Even Catchier Title")

    image = Image.fromarray(np.random.randint(12, 14, size=(40, 10, 3)), mode="RGB")
    page = layout.PageLayout(
        number=0, image=image, layout=doc_layout, model=MockLayoutModel(doc_layout)
    )
    page.get_elements_with_model()

    assert str(page) == "\n\nAn Even Catchier Title"


def test_read_pdf(monkeypatch, mock_page_layout):
    image = np.random.randint(12, 24, (40, 40))
    images = [image, image]

    layouts = [mock_page_layout, mock_page_layout]

    monkeypatch.setattr(
        models, "UnstructuredDetectronModel", partial(MockLayoutModel, layout=mock_page_layout)
    )
    monkeypatch.setattr(detectron2, "is_detectron2_available", lambda *args: True)

    with patch.object(layout, "load_pdf", return_value=(layouts, images)):
        doc = layout.DocumentLayout.from_file("fake-file.pdf")

        assert str(doc).startswith("A Catchy Title")
        assert str(doc).count("A Catchy Title") == 2  # Once for each page
        assert str(doc).endswith("A very repetitive narrative. ")

        assert doc.pages[0].elements[0].to_dict()["text"] == "A Catchy Title"

        pages = doc.pages
        assert str(doc) == "\n\n".join([str(page) for page in pages])


@pytest.mark.parametrize("model_name", [None, "checkbox", "fake"])
def test_process_data_with_model(monkeypatch, mock_page_layout, model_name):
    monkeypatch.setattr(layout, "get_model", lambda x: MockLayoutModel(mock_page_layout))
    monkeypatch.setattr(
        layout.DocumentLayout,
        "from_file",
        lambda *args, **kwargs: layout.DocumentLayout.from_pages([]),
    )
    with patch("builtins.open", mock_open(read_data=b"000000")):
        assert layout.process_data_with_model(open(""), model_name=model_name)


def test_process_data_with_model_raises_on_invalid_model_name():
    with patch("builtins.open", mock_open(read_data=b"000000")):
        with pytest.raises(models.UnknownModelException):
            layout.process_data_with_model(open(""), model_name="fake")


@pytest.mark.parametrize("model_name", [None, "checkbox"])
def test_process_file_with_model(monkeypatch, mock_page_layout, model_name):
    def mock_initialize(self, *args, **kwargs):
        self.model = MockLayoutModel(mock_page_layout)

    monkeypatch.setattr(models, "get_model", lambda x: MockLayoutModel(mock_page_layout))
    monkeypatch.setattr(
        layout.DocumentLayout,
        "from_file",
        lambda *args, **kwargs: layout.DocumentLayout.from_pages([]),
    )
    monkeypatch.setattr(models.UnstructuredDetectronModel, "initialize", mock_initialize)
    filename = ""
    assert layout.process_file_with_model(filename, model_name=model_name)


def test_process_file_with_model_raises_on_invalid_model_name():
    with pytest.raises(models.UnknownModelException):
        layout.process_file_with_model("", model_name="fake")


class MockPoints:
    def tolist(self):
        return [1, 2, 3, 4]


class MockEmbeddedTextRegion(layout.EmbeddedTextRegion):
    def __init__(self, type=None, text=None, ocr_text=None):
        self.type = type
        self.text = text
        self.ocr_text = ocr_text

    @property
    def points(self):
        return MockPoints()


class MockPageLayout(layout.PageLayout):
    def __init__(
        self,
        layout=None,
        model=None,
        ocr_strategy="auto",
        ocr_languages="eng",
        extract_tables=False,
    ):
        self.image = None
        self.layout = layout
        self.model = model
        self.ocr_strategy = ocr_strategy
        self.ocr_languages = ocr_languages
        self.extract_tables = extract_tables

    def ocr(self, text_block: MockEmbeddedTextRegion):
        return text_block.ocr_text


@pytest.mark.parametrize(
    "text, expected",
    [("base", 0.0), ("", 0.0), ("(cid:2)", 1.0), ("(cid:1)a", 0.5), ("c(cid:1)ab", 0.25)],
)
def test_cid_ratio(text, expected):
    assert elements.cid_ratio(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("base", False), ("(cid:2)", True), ("(cid:1234567890)", True), ("jkl;(cid:12)asdf", True)],
)
def test_is_cid_present(text, expected):
    assert elements.is_cid_present(text) == expected


class MockLayout:
    def __init__(self, *elements):
        self.elements = elements

    def __len__(self):
        return len(self.elements)

    def sort(self, key, inplace):
        return self.elements

    def __iter__(self):
        return iter(self.elements)

    def get_texts(self):
        return [el.text for el in self.elements]

    def filter_by(self, *args, **kwargs):
        return MockLayout()


@pytest.mark.parametrize(
    "block_text, layout_texts, expected_text",
    [
        ("no ocr", ["pieced", "together", "group"], "no ocr"),
        (None, ["pieced", "together", "group"], "pieced together group"),
    ],
)
def test_get_element_from_block(block_text, layout_texts, mock_image, expected_text):
    with patch("unstructured_inference.inference.elements.ocr", return_value="ocr"):
        block = layout.TextRegion(0, 0, 10, 10, text=block_text)
        captured_layout = [
            layout.TextRegion(i + 1, i + 1, i + 2, i + 2, text=text)
            for i, text in enumerate(layout_texts)
        ]
        assert (
            layout.get_element_from_block(block, mock_image, captured_layout).text == expected_text
        )


def test_get_elements_from_block_raises():
    with pytest.raises(ValueError):
        block = layout.TextRegion(0, 0, 10, 10, text=None)
        layout.get_element_from_block(block, None, None)


@pytest.mark.parametrize("filetype", ("png", "jpg"))
def test_from_image_file(monkeypatch, mock_page_layout, filetype):
    def mock_get_elements(self, *args, **kwargs):
        self.elements = [mock_page_layout]

    monkeypatch.setattr(layout.PageLayout, "get_elements_with_model", mock_get_elements)
    elements = (
        layout.DocumentLayout.from_image_file(f"sample-docs/loremipsum.{filetype}")
        .pages[0]
        .elements
    )
    assert elements[0] == mock_page_layout


def test_from_image_file_raises_with_empty_fn():
    with pytest.raises(FileNotFoundError):
        layout.DocumentLayout.from_image_file("")


def test_from_image_file_raises_isadirectoryerror_with_dir():
    with tempfile.TemporaryDirectory() as tempdir, pytest.raises(IsADirectoryError):
        layout.DocumentLayout.from_image_file(tempdir)


def test_from_file_raises_on_length_mismatch(monkeypatch):
    monkeypatch.setattr(layout, "load_pdf", lambda *args, **kwargs: ([None, None], []))
    with pytest.raises(RuntimeError) as e:
        layout.DocumentLayout.from_file("fake_file")
    assert "poppler" in str(e).lower()


@pytest.mark.parametrize("idx", range(2))
def test_get_elements_from_layout(mock_page_layout, idx):
    page = MockPageLayout(layout=mock_page_layout)
    block = mock_page_layout[idx].pad(3)
    fixed_layout = [block]
    elements = page.get_elements_from_layout(fixed_layout)
    assert elements[0].text == block.text


@pytest.mark.parametrize(
    "fixed_layouts, called_method, not_called_method",
    [
        ([MockLayout()], "get_elements_from_layout", "get_elements_with_model"),
        (None, "get_elements_with_model", "get_elements_from_layout"),
    ],
)
def test_from_file_fixed_layout(fixed_layouts, called_method, not_called_method):
    with patch.object(layout.PageLayout, "get_elements_with_model", return_value=[]), patch.object(
        layout.PageLayout, "get_elements_from_layout", return_value=[]
    ):
        layout.DocumentLayout.from_file("sample-docs/loremipsum.pdf", fixed_layouts=fixed_layouts)
        getattr(layout.PageLayout, called_method).assert_called()
        getattr(layout.PageLayout, not_called_method).assert_not_called()


def test_invalid_ocr_strategy_raises(mock_image):
    with pytest.raises(ValueError):
        layout.PageLayout(0, mock_image, MockLayout(), ocr_strategy="fake_strategy")


@pytest.mark.parametrize(
    ("text", "expected"), [("a\ts\x0cd\nfas\fd\rf\b", "asdfasdf"), ("\"'\\", "\"'\\")]
)
def test_remove_control_characters(text, expected):
    assert elements.remove_control_characters(text) == expected


no_text_region = layout.EmbeddedTextRegion(0, 0, 100, 100)
text_region = layout.EmbeddedTextRegion(0, 0, 100, 100, text="test")
cid_text_region = layout.EmbeddedTextRegion(
    0, 0, 100, 100, text="(cid:1)(cid:2)(cid:3)(cid:4)(cid:5)"
)
overlapping_rect = layout.ImageTextRegion(50, 50, 150, 150)
nonoverlapping_rect = layout.ImageTextRegion(150, 150, 200, 200)
populated_text_region = layout.EmbeddedTextRegion(50, 50, 60, 60, text="test")
unpopulated_text_region = layout.EmbeddedTextRegion(50, 50, 60, 60, text=None)


@pytest.mark.parametrize(
    ("region", "objects", "ocr_strategy", "expected"),
    [
        (no_text_region, [nonoverlapping_rect], "auto", False),
        (no_text_region, [overlapping_rect], "auto", True),
        (no_text_region, [], "auto", False),
        (no_text_region, [populated_text_region, nonoverlapping_rect], "auto", False),
        (no_text_region, [populated_text_region, overlapping_rect], "auto", False),
        (no_text_region, [populated_text_region], "auto", False),
        (no_text_region, [unpopulated_text_region, nonoverlapping_rect], "auto", False),
        (no_text_region, [unpopulated_text_region, overlapping_rect], "auto", True),
        (no_text_region, [unpopulated_text_region], "auto", False),
        *list(
            product(
                [text_region],
                [
                    [],
                    [populated_text_region],
                    [unpopulated_text_region],
                    [nonoverlapping_rect],
                    [overlapping_rect],
                    [populated_text_region, nonoverlapping_rect],
                    [populated_text_region, overlapping_rect],
                    [unpopulated_text_region, nonoverlapping_rect],
                    [unpopulated_text_region, overlapping_rect],
                ],
                ["auto"],
                [False],
            )
        ),
        *list(
            product(
                [cid_text_region],
                [
                    [],
                    [populated_text_region],
                    [unpopulated_text_region],
                    [overlapping_rect],
                    [populated_text_region, overlapping_rect],
                    [unpopulated_text_region, overlapping_rect],
                ],
                ["auto"],
                [True],
            )
        ),
        *list(
            product(
                [no_text_region, text_region, cid_text_region],
                [
                    [],
                    [populated_text_region],
                    [unpopulated_text_region],
                    [nonoverlapping_rect],
                    [overlapping_rect],
                    [populated_text_region, nonoverlapping_rect],
                    [populated_text_region, overlapping_rect],
                    [unpopulated_text_region, nonoverlapping_rect],
                    [unpopulated_text_region, overlapping_rect],
                ],
                ["force"],
                [True],
            )
        ),
        *list(
            product(
                [no_text_region, text_region, cid_text_region],
                [
                    [],
                    [populated_text_region],
                    [unpopulated_text_region],
                    [nonoverlapping_rect],
                    [overlapping_rect],
                    [populated_text_region, nonoverlapping_rect],
                    [populated_text_region, overlapping_rect],
                    [unpopulated_text_region, nonoverlapping_rect],
                    [unpopulated_text_region, overlapping_rect],
                ],
                ["never"],
                [False],
            )
        ),
    ],
)
def test_ocr_image(region, objects, ocr_strategy, expected):
    assert elements.needs_ocr(region, objects, ocr_strategy) is expected


def test_load_pdf():
    layouts, images = layout.load_pdf("sample-docs/loremipsum.pdf")
    assert len(layouts)
    assert len(images)
    assert len(layouts) == len(images)


def test_load_pdf_with_images():
    layouts, _ = layout.load_pdf("sample-docs/loremipsum-flat.pdf")
    first_page_layout = layouts[0]
    assert any(isinstance(obj, layout.ImageTextRegion) for obj in first_page_layout)


def test_load_pdf_image_placement():
    layouts, images = layout.load_pdf("sample-docs/layout-parser-paper.pdf")
    page_layout = layouts[5]
    image_regions = [region for region in page_layout if isinstance(region, layout.ImageTextRegion)]
    image_region = image_regions[0]
    # Image is in top half of the page, so that should be reflected in the pixel coordinates
    assert image_region.y1 < images[5].height / 2
    assert image_region.y2 < images[5].height / 2
