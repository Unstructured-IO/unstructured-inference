import os.path
import tempfile
from functools import partial
from itertools import product
from unittest.mock import mock_open, patch

import numpy as np
import pytest
from PIL import Image

import unstructured_inference.models.base as models
from unstructured_inference.inference import elements, layout, layoutelement
from unstructured_inference.inference.layout import create_image_output_dir
from unstructured_inference.models import detectron2, tesseract
from unstructured_inference.models.unstructuredmodel import (
    UnstructuredElementExtractionModel,
    UnstructuredObjectDetectionModel,
)


@pytest.fixture()
def mock_image():
    return Image.new("1", (1, 1))


@pytest.fixture()
def mock_initial_layout():
    text_block = layout.EmbeddedTextRegion(2, 4, 6, 8, text="A very repetitive narrative. " * 10)

    title_block = layout.EmbeddedTextRegion(1, 2, 3, 4, text="A Catchy Title")

    return [text_block, title_block]


@pytest.fixture()
def mock_final_layout():
    text_block = layoutelement.LayoutElement(
        2,
        4,
        6,
        8,
        text="A very repetitive narrative. " * 10,
        type="NarrativeText",
    )

    title_block = layoutelement.LayoutElement(1, 2, 3, 4, text="A Catchy Title", type="Title")

    return [text_block, title_block]


def test_pdf_page_converts_images_to_array(mock_image):
    def verify_image_array():
        assert page.image_array is None
        image_array = page._get_image_array()
        assert isinstance(image_array, np.ndarray)
        assert page.image_array.all() == image_array.all()

    # Scenario 1: where self.image exists
    page = layout.PageLayout(number=0, image=mock_image, layout=[])
    verify_image_array()

    # Scenario 2: where self.image is None, but self.image_path exists
    page.image_array = None
    page.image = None
    page.image_path = "fake-image-path"
    with patch.object(Image, "open", return_value=mock_image):
        verify_image_array()


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


def test_get_page_elements(monkeypatch, mock_final_layout):
    image = np.random.randint(12, 24, (40, 40))
    page = layout.PageLayout(
        number=0,
        image=image,
        layout=mock_final_layout,
        detection_model=MockLayoutModel(mock_final_layout),
    )

    elements = page.get_elements_with_detection_model(inplace=False)

    assert str(elements[0]) == "A Catchy Title"
    assert str(elements[1]).startswith("A very repetitive narrative.")

    page.get_elements_with_detection_model(inplace=True)
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
    doc_initial_layout = [text_block, image_block]
    text_layoutelement = layoutelement.LayoutElement(
        2,
        4,
        6,
        8,
        text=None,
        type="UncategorizedText",
    )
    image_layoutelement = layoutelement.LayoutElement(8, 14, 16, 18, text=None, type="Image")
    doc_final_layout = [text_layoutelement, image_layoutelement]

    monkeypatch.setattr(detectron2, "is_detectron2_available", lambda *args: True)
    monkeypatch.setattr(elements, "ocr", lambda *args, **kwargs: "An Even Catchier Title")

    image = Image.fromarray(np.random.randint(12, 14, size=(40, 10, 3)), mode="RGB")
    page = layout.PageLayout(
        number=0,
        image=image,
        layout=doc_initial_layout,
        detection_model=MockLayoutModel(doc_final_layout),
    )
    page.get_elements_with_detection_model()

    assert str(page) == "\n\nAn Even Catchier Title"


def test_read_pdf(monkeypatch, mock_initial_layout, mock_final_layout, mock_image):
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path1 = os.path.join(tmpdir, "mock1.jpg")
        image_path2 = os.path.join(tmpdir, "mock2.jpg")
        mock_image.save(image_path1)
        mock_image.save(image_path2)
        image_paths = [image_path1, image_path2]

        layouts = [mock_initial_layout, mock_initial_layout]

        monkeypatch.setattr(
            models,
            "UnstructuredDetectronModel",
            partial(MockLayoutModel, layout=mock_final_layout),
        )
        monkeypatch.setattr(detectron2, "is_detectron2_available", lambda *args: True)

        with patch.object(layout, "load_pdf", return_value=(layouts, image_paths)):
            model = layout.get_model("detectron2_lp")
            doc = layout.DocumentLayout.from_file("fake-file.pdf", detection_model=model)

            assert str(doc).startswith("A Catchy Title")
            assert str(doc).count("A Catchy Title") == 2  # Once for each page
            assert str(doc).endswith("A very repetitive narrative. ")

            assert doc.pages[0].elements[0].to_dict()["text"] == "A Catchy Title"

            pages = doc.pages
            assert str(doc) == "\n\n".join([str(page) for page in pages])


@pytest.mark.parametrize("model_name", [None, "checkbox", "fake"])
def test_process_data_with_model(monkeypatch, mock_final_layout, model_name):
    monkeypatch.setattr(layout, "get_model", lambda x: MockLayoutModel(mock_final_layout))
    monkeypatch.setattr(
        layout.DocumentLayout,
        "from_file",
        lambda *args, **kwargs: layout.DocumentLayout.from_pages([]),
    )

    def new_isinstance(obj, cls):
        if type(obj) == MockLayoutModel:
            return True
        else:
            return isinstance(obj, cls)

    with patch("builtins.open", mock_open(read_data=b"000000")), patch(
        "unstructured_inference.inference.layout.UnstructuredObjectDetectionModel",
        MockLayoutModel,
    ), open("") as fp:
        assert layout.process_data_with_model(fp, model_name=model_name)


def test_process_data_with_model_raises_on_invalid_model_name():
    with patch("builtins.open", mock_open(read_data=b"000000")), pytest.raises(
        models.UnknownModelException,
    ), open("") as fp:
        layout.process_data_with_model(fp, model_name="fake")


@pytest.mark.parametrize("model_name", [None, "checkbox"])
def test_process_file_with_model(monkeypatch, mock_final_layout, model_name):
    def mock_initialize(self, *args, **kwargs):
        self.model = MockLayoutModel(mock_final_layout)

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
    ("text", "expected"),
    [("base", 0.0), ("", 0.0), ("(cid:2)", 1.0), ("(cid:1)a", 0.5), ("c(cid:1)ab", 0.25)],
)
def test_cid_ratio(text, expected):
    assert elements.cid_ratio(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
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
    ("block_text", "layout_texts", "expected_text"),
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


@pytest.mark.parametrize("filetype", ["png", "jpg"])
def test_from_image_file(monkeypatch, mock_final_layout, filetype):
    def mock_get_elements(self, *args, **kwargs):
        self.elements = [mock_final_layout]

    monkeypatch.setattr(layout.PageLayout, "get_elements_with_detection_model", mock_get_elements)
    filename = f"sample-docs/loremipsum.{filetype}"
    image = Image.open(filename)
    image_metadata = {
        "format": image.format,
        "width": image.width,
        "height": image.height,
    }

    doc = layout.DocumentLayout.from_image_file(filename)
    page = doc.pages[0]
    assert page.elements[0] == mock_final_layout
    assert page.image is None
    assert page.image_path == os.path.abspath(filename)
    assert page.image_metadata == image_metadata


def test_from_file(monkeypatch, mock_final_layout):
    def mock_get_elements(self, *args, **kwargs):
        self.elements = [mock_final_layout]

    monkeypatch.setattr(layout.PageLayout, "get_elements_with_detection_model", mock_get_elements)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "loremipsum.ppm")
        image = Image.open('sample-docs/loremipsum.jpg')
        image.save(image_path)
        image_metadata = {
            "format": 'PPM',
            "width": image.width,
            "height": image.height,
        }

        with patch.object(
            layout,
            "create_image_output_dir",
            return_value=tmpdir,
        ), patch.object(
            layout,
            "load_pdf",
            lambda *args, **kwargs: ([[]], [image_path]),
        ):
            doc = layout.DocumentLayout.from_file("fake-file.pdf")
            page = doc.pages[0]
            assert page.elements[0] == mock_final_layout
            assert page.image_metadata == image_metadata
            assert page.image_path == image_path
            assert page.image is None


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
def test_get_elements_from_layout(mock_initial_layout, idx):
    page = MockPageLayout(layout=mock_initial_layout)
    block = mock_initial_layout[idx].pad(3)
    fixed_layout = [block]
    elements = page.get_elements_from_layout(fixed_layout)
    assert elements[0].text == block.text


def test_page_numbers_in_page_objects():
    with patch(
        "unstructured_inference.inference.layout.PageLayout.get_elements_with_detection_model",
    ) as mock_get_elements:
        doc = layout.DocumentLayout.from_file("sample-docs/layout-parser-paper.pdf")
        mock_get_elements.assert_called()
        assert [page.number for page in doc.pages] == list(range(1, len(doc.pages) + 1))


@pytest.mark.parametrize(
    ("fixed_layouts", "called_method", "not_called_method"),
    [
        ([MockLayout()], "get_elements_from_layout", "get_elements_with_detection_model"),
        (None, "get_elements_with_detection_model", "get_elements_from_layout"),
    ],
)
def test_from_file_fixed_layout(fixed_layouts, called_method, not_called_method):
    with patch.object(
        layout.PageLayout,
        "get_elements_with_detection_model",
        return_value=[],
    ), patch.object(
        layout.PageLayout,
        "get_elements_from_layout",
        return_value=[],
    ):
        layout.DocumentLayout.from_file("sample-docs/loremipsum.pdf", fixed_layouts=fixed_layouts)
        getattr(layout.PageLayout, called_method).assert_called()
        getattr(layout.PageLayout, not_called_method).assert_not_called()


def test_invalid_ocr_strategy_raises(mock_image):
    with pytest.raises(ValueError):
        layout.PageLayout(0, mock_image, MockLayout(), ocr_strategy="fake_strategy")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("a\ts\x0cd\nfas\fd\rf\b", "asdfasdf"), ("\"'\\", "\"'\\")],
)
def test_remove_control_characters(text, expected):
    assert elements.remove_control_characters(text) == expected


no_text_region = layout.EmbeddedTextRegion(0, 0, 100, 100)
text_region = layout.EmbeddedTextRegion(0, 0, 100, 100, text="test")
cid_text_region = layout.EmbeddedTextRegion(
    0,
    0,
    100,
    100,
    text="(cid:1)(cid:2)(cid:3)(cid:4)(cid:5)",
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
            ),
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
            ),
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
            ),
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
            ),
        ),
    ],
)
def test_ocr_image(region, objects, ocr_strategy, expected):
    assert elements.needs_ocr(region, objects, ocr_strategy) is expected


@pytest.mark.parametrize("filename", ["loremipsum.pdf", "IRS-form-1987.pdf"])
def test_load_pdf(filename):
    layouts, images = layout.load_pdf(f"sample-docs/{filename}")
    assert len(layouts)
    for lo in layouts:
        assert len(lo)
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


def test_load_pdf_raises_with_path_only_no_output_folder():
    with pytest.raises(ValueError):
        layout.load_pdf(
            "sample-docs/loremipsum-flat.pdf",
            path_only=True,
        )


@pytest.mark.skip("Temporarily removed multicolumn to fix ordering")
def test_load_pdf_with_multicolumn_layout_and_ocr(filename="sample-docs/design-thinking.pdf"):
    layouts, images = layout.load_pdf(filename)
    doc = layout.process_file_with_model(filename=filename, model_name=None)
    test_snippets = ["Key to design thinking", "Design thinking also", "But in recent years"]

    test_elements = []
    for element in doc.pages[0].elements:
        for snippet in test_snippets:
            if element.text.startswith(snippet):
                test_elements.append(element)

    for i, element in enumerate(test_elements):
        assert element.text.startswith(test_snippets[i])


@pytest.mark.parametrize("colors", ["red", None])
def test_annotate(colors):
    test_image_arr = np.ones((100, 100, 3), dtype="uint8")
    image = Image.fromarray(test_image_arr)
    page = layout.PageLayout(number=1, image=image, layout=None)
    coords1 = (21, 30, 37, 41)
    rect1 = elements.Rectangle(*coords1)
    coords2 = (1, 10, 7, 11)
    rect2 = elements.Rectangle(*coords2)
    page.elements = [rect1, rect2]
    annotated_image = page.annotate(colors=colors)
    annotated_array = np.array(annotated_image)
    for x1, y1, x2, y2 in [coords1, coords2]:
        # Make sure the pixels on the edge of the box are red
        for i, expected in zip(range(3), [255, 0, 0]):
            assert all(annotated_array[y1, x1:x2, i] == expected)
            assert all(annotated_array[y2, x1:x2, i] == expected)
            assert all(annotated_array[y1:y2, x1, i] == expected)
            assert all(annotated_array[y1:y2, x2, i] == expected)
        # Make sure almost all the pixels are not changed
        assert ((annotated_array[:, :, 0] == 1).mean()) > 0.992
        assert ((annotated_array[:, :, 1] == 1).mean()) > 0.992
        assert ((annotated_array[:, :, 2] == 1).mean()) > 0.992


def test_textregion_returns_empty_ocr_never(mock_image):
    tr = elements.TextRegion(0, 0, 24, 24)
    assert tr.extract_text(objects=None, image=mock_image, ocr_strategy="never") == ""


@pytest.mark.parametrize(("text", "expected"), [("asdf", "asdf"), (None, "")])
def test_embedded_text_region(text, expected):
    etr = elements.EmbeddedTextRegion(0, 0, 24, 24, text=text)
    assert etr.extract_text(objects=None) == expected


@pytest.mark.parametrize(
    ("text", "ocr_strategy", "expected"),
    [
        (None, "never", ""),
        (None, "always", "asdf"),
        ("i have text", "never", "i have text"),
        ("i have text", "always", "i have text"),
    ],
)
def test_image_text_region(text, ocr_strategy, expected, mock_image):
    itr = elements.ImageTextRegion(0, 0, 24, 24, text=text)
    with patch.object(elements, "ocr", return_value="asdf"):
        assert (
            itr.extract_text(objects=None, image=mock_image, ocr_strategy=ocr_strategy) == expected
        )


@pytest.fixture()
def ordering_layout():
    elements = [
        layout.LayoutElement(x1=447.0, y1=315.0, x2=1275.7, y2=413.0, text="0"),
        layout.LayoutElement(x1=380.6, y1=473.4, x2=1334.8, y2=533.9, text="1"),
        layout.LayoutElement(x1=578.6, y1=556.8, x2=1109.0, y2=874.4, text="2"),
        layout.LayoutElement(x1=444.5, y1=942.3, x2=1261.1, y2=1584.1, text="3"),
        layout.LayoutElement(x1=444.8, y1=1609.4, x2=1257.2, y2=1665.2, text="4"),
        layout.LayoutElement(x1=414.0, y1=1718.8, x2=635.0, y2=1755.2, text="5"),
        layout.LayoutElement(x1=372.6, y1=1786.9, x2=1333.6, y2=1848.7, text="6"),
    ]
    return elements


def test_layout_order(mock_image, ordering_layout):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_image_path = os.path.join(tmpdir, "mock.jpg")
        mock_image.save(mock_image_path)
        with patch.object(layout, "get_model", lambda: lambda x: ordering_layout), patch.object(
            layout,
            "load_pdf",
            lambda *args, **kwargs: ([[]], [mock_image_path]),
        ), patch.object(
            layout,
            "UnstructuredObjectDetectionModel",
            object,
        ):
            doc = layout.DocumentLayout.from_file("sample-docs/layout-parser-paper.pdf")
            page = doc.pages[0]
    for n, element in enumerate(page.elements):
        assert element.text == str(n)


def test_page_layout_raises_when_multiple_models_passed(mock_image, mock_initial_layout):
    with pytest.raises(ValueError):
        layout.PageLayout(
            0,
            mock_image,
            mock_initial_layout,
            detection_model="something",
            element_extraction_model="something else",
        )


class MockElementExtractionModel:
    def __call__(self, x):
        return [1, 2, 3]


@pytest.mark.parametrize(("inplace", "expected"), [(True, None), (False, [1, 2, 3])])
def test_get_elements_using_image_extraction(mock_image, inplace, expected):
    page = layout.PageLayout(
        1,
        mock_image,
        None,
        element_extraction_model=MockElementExtractionModel(),
    )
    assert page.get_elements_using_image_extraction(inplace=inplace) == expected


def test_get_elements_using_image_extraction_raises_with_no_extraction_model(mock_image):
    page = layout.PageLayout(1, mock_image, None, element_extraction_model=None)
    with pytest.raises(ValueError):
        page.get_elements_using_image_extraction()


def test_get_elements_with_detection_model_raises_with_wrong_default_model(monkeypatch):
    monkeypatch.setattr(layout, "get_model", lambda *x: MockLayoutModel(mock_final_layout))
    page = layout.PageLayout(1, mock_image, None)
    with pytest.raises(NotImplementedError):
        page.get_elements_with_detection_model()


@pytest.mark.parametrize(
    (
        "detection_model",
        "element_extraction_model",
        "detection_model_called",
        "element_extraction_model_called",
    ),
    [(None, "asdf", False, True), ("asdf", None, True, False)],
)
def test_from_image(
    mock_image,
    detection_model,
    element_extraction_model,
    detection_model_called,
    element_extraction_model_called,
):
    with patch.object(
        layout.PageLayout,
        "get_elements_using_image_extraction",
    ) as mock_image_extraction, patch.object(
        layout.PageLayout,
        "get_elements_with_detection_model",
    ) as mock_detection:
        layout.PageLayout.from_image(
            mock_image,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
        )
        assert mock_image_extraction.called == element_extraction_model_called
        assert mock_detection.called == detection_model_called


class MockUnstructuredElementExtractionModel(UnstructuredElementExtractionModel):
    def initialize(self, *args, **kwargs):
        return super().initialize(*args, **kwargs)

    def predict(self, x: Image):
        return super().predict(x)


class MockUnstructuredDetectionModel(UnstructuredObjectDetectionModel):
    def initialize(self, *args, **kwargs):
        return super().initialize(*args, **kwargs)

    def predict(self, x: Image):
        return super().predict(x)


@pytest.mark.parametrize(
    ("model_type", "is_detection_model"),
    [
        (MockUnstructuredElementExtractionModel, False),
        (MockUnstructuredDetectionModel, True),
    ],
)
def test_process_file_with_model_routing(monkeypatch, model_type, is_detection_model):
    model = model_type()
    monkeypatch.setattr(layout, "get_model", lambda *x: model)
    with patch.object(layout.DocumentLayout, "from_file") as mock_from_file:
        layout.process_file_with_model("asdf", model_name="fake", is_image=False)
        if is_detection_model:
            detection_model = model
            element_extraction_model = None
        else:
            detection_model = None
            element_extraction_model = model
        mock_from_file.assert_called_once_with(
            "asdf",
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
            ocr_strategy="auto",
            ocr_languages="eng",
            fixed_layouts=None,
            extract_tables=False,
            pdf_image_dpi=200,
        )


@pytest.mark.parametrize(("pdf_image_dpi", "expected"), [(200, 2200), (100, 1100)])
def test_exposed_pdf_image_dpi(pdf_image_dpi, expected, monkeypatch):
    with patch.object(layout.PageLayout, "from_image") as mock_from_image:
        layout.DocumentLayout.from_file("sample-docs/loremipsum.pdf", pdf_image_dpi=pdf_image_dpi)
        assert mock_from_image.call_args[0][0].height == expected


def test_create_image_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_f_path = os.path.join(tmpdir, "loremipsum.pdf")
        output_dir = create_image_output_dir(tmp_f_path)
        expected_output_dir = os.path.join(os.path.abspath(tmpdir), "loremipsum")
        assert os.path.isdir(output_dir)
        assert os.path.isabs(output_dir)
        assert output_dir == expected_output_dir



