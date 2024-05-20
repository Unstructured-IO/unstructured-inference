import os
import os.path
import tempfile
from unittest.mock import mock_open, patch

import numpy as np
import pytest
from PIL import Image

import unstructured_inference.models.base as models
from unstructured_inference.inference import elements, layout, layoutelement
from unstructured_inference.inference.elements import (
    EmbeddedTextRegion,
    ImageTextRegion,
)
from unstructured_inference.models.unstructuredmodel import (
    UnstructuredElementExtractionModel,
    UnstructuredObjectDetectionModel,
)

skip_outside_ci = os.getenv("CI", "").lower() in {"", "false", "f", "0"}


@pytest.fixture()
def mock_image():
    return Image.new("1", (1, 1))


@pytest.fixture()
def mock_initial_layout():
    text_block = EmbeddedTextRegion.from_coords(
        2,
        4,
        6,
        8,
        text="A very repetitive narrative. " * 10,
        source="Mock",
    )

    title_block = EmbeddedTextRegion.from_coords(
        1,
        2,
        3,
        4,
        text="A Catchy Title",
        source="Mock",
    )

    return [text_block, title_block]


@pytest.fixture()
def mock_final_layout():
    text_block = layoutelement.LayoutElement.from_coords(
        2,
        4,
        6,
        8,
        source="Mock",
        text="A very repetitive narrative. " * 10,
        type="NarrativeText",
    )

    title_block = layoutelement.LayoutElement.from_coords(
        1,
        2,
        3,
        4,
        source="Mock",
        text="A Catchy Title",
        type="Title",
    )

    return [text_block, title_block]


def test_pdf_page_converts_images_to_array(mock_image):
    def verify_image_array():
        assert page.image_array is None
        image_array = page._get_image_array()
        assert isinstance(image_array, np.ndarray)
        assert page.image_array.all() == image_array.all()

    # Scenario 1: where self.image exists
    page = layout.PageLayout(number=0, image=mock_image)
    verify_image_array()

    # Scenario 2: where self.image is None, but self.image_path exists
    page.image_array = None
    page.image = None
    page.image_path = "mock_path_to_image"
    with patch.object(Image, "open", return_value=mock_image):
        verify_image_array()


class MockLayoutModel:
    def __init__(self, layout):
        self.layout_return = layout

    def __call__(self, *args):
        return self.layout_return

    def initialize(self, *args, **kwargs):
        pass

    def deduplicate_detected_elements(self, elements, *args, **kwargs):
        return elements


def test_get_page_elements(monkeypatch, mock_final_layout):
    image = Image.fromarray(np.random.randint(12, 14, size=(40, 10, 3)), mode="RGB")
    page = layout.PageLayout(
        number=0,
        image=image,
        detection_model=MockLayoutModel(mock_final_layout),
    )
    elements = page.get_elements_with_detection_model(inplace=False)
    page.get_elements_with_detection_model(inplace=True)
    assert elements == page.elements


class MockPool:
    def map(self, f, xs):
        return [f(x) for x in xs]

    def close(self):
        pass

    def join(self):
        pass


@pytest.mark.parametrize("model_name", [None, "checkbox", "fake"])
def test_process_data_with_model(monkeypatch, mock_final_layout, model_name):
    monkeypatch.setattr(layout, "get_model", lambda x: MockLayoutModel(mock_final_layout))
    monkeypatch.setattr(
        layout.DocumentLayout,
        "from_file",
        lambda *args, **kwargs: layout.DocumentLayout.from_pages([]),
    )

    def new_isinstance(obj, cls):
        if type(obj) is MockLayoutModel:
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


def test_process_file_no_warnings(monkeypatch, mock_final_layout, recwarn):
    def mock_initialize(self, *args, **kwargs):
        self.model = MockLayoutModel(mock_final_layout)

    monkeypatch.setattr(
        layout.DocumentLayout,
        "from_file",
        lambda *args, **kwargs: layout.DocumentLayout.from_pages([]),
    )
    monkeypatch.setattr(models.UnstructuredDetectronModel, "initialize", mock_initialize)
    filename = ""
    layout.process_file_with_model(filename, model_name=None)
    # There should be no UserWarning, but if there is one it should not have the following message
    with pytest.raises(AssertionError, match="not found in warning list"):
        user_warning = recwarn.pop(UserWarning)
        assert "not in available provider names" not in str(user_warning.message)


def test_process_file_with_model_raises_on_invalid_model_name():
    with pytest.raises(models.UnknownModelException):
        layout.process_file_with_model("", model_name="fake")


class MockPoints:
    def tolist(self):
        return [1, 2, 3, 4]


class MockEmbeddedTextRegion(EmbeddedTextRegion):
    def __init__(self, type=None, text=None):
        self.type = type
        self.text = text

    @property
    def points(self):
        return MockPoints()


class MockPageLayout(layout.PageLayout):
    def __init__(
        self,
        number=1,
        image=None,
        model=None,
        detection_model=None,
    ):
        self.image = image
        self.layout = layout
        self.model = model
        self.number = number
        self.detection_model = detection_model


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


@pytest.mark.parametrize("element_extraction_model", [None, "foo"])
@pytest.mark.parametrize("filetype", ["png", "jpg", "tiff"])
def test_from_image_file(monkeypatch, mock_final_layout, filetype, element_extraction_model):
    def mock_get_elements(self, *args, **kwargs):
        self.elements = [mock_final_layout]

    monkeypatch.setattr(layout.PageLayout, "get_elements_with_detection_model", mock_get_elements)
    monkeypatch.setattr(layout.PageLayout, "get_elements_using_image_extraction", mock_get_elements)
    filename = f"sample-docs/loremipsum.{filetype}"
    image = Image.open(filename)
    image_metadata = {
        "format": image.format,
        "width": image.width,
        "height": image.height,
    }

    doc = layout.DocumentLayout.from_image_file(
        filename,
        element_extraction_model=element_extraction_model,
    )
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
        image = Image.open("sample-docs/loremipsum.jpg")
        image.save(image_path)
        image_metadata = {
            "format": "PPM",
            "width": image.width,
            "height": image.height,
        }

        with patch.object(
            layout,
            "convert_pdf_to_image",
            lambda *args, **kwargs: ([image_path]),
        ):
            doc = layout.DocumentLayout.from_file("fake-file.pdf")
            page = doc.pages[0]
            assert page.elements[0] == mock_final_layout
            assert page.image_metadata == image_metadata
            assert page.image is None


def test_from_image_file_raises_with_empty_fn():
    with pytest.raises(FileNotFoundError):
        layout.DocumentLayout.from_image_file("")


def test_from_image_file_raises_isadirectoryerror_with_dir():
    with tempfile.TemporaryDirectory() as tempdir, pytest.raises(IsADirectoryError):
        layout.DocumentLayout.from_image_file(tempdir)


@pytest.mark.parametrize("idx", range(2))
def test_get_elements_from_layout(mock_initial_layout, idx):
    page = MockPageLayout()
    block = mock_initial_layout[idx]
    block.bbox.pad(3)
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
        (
            [MockLayout()],
            "get_elements_from_layout",
            "get_elements_with_detection_model",
        ),
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


no_text_region = EmbeddedTextRegion.from_coords(0, 0, 100, 100)
text_region = EmbeddedTextRegion.from_coords(0, 0, 100, 100, text="test")
overlapping_rect = ImageTextRegion.from_coords(50, 50, 150, 150)
nonoverlapping_rect = ImageTextRegion.from_coords(150, 150, 200, 200)
populated_text_region = EmbeddedTextRegion.from_coords(50, 50, 60, 60, text="test")
unpopulated_text_region = EmbeddedTextRegion.from_coords(50, 50, 60, 60, text=None)


@pytest.mark.parametrize(
    ("colors", "add_details", "threshold"),
    [("red", False, 0.992), (None, False, 0.992), ("red", True, 0.8)],
)
def test_annotate(colors, add_details, threshold):
    def check_annotated_image():
        annotated_array = np.array(annotated_image)
        for coords in [coords1, coords2]:
            x1, y1, x2, y2 = coords
            # Make sure the pixels on the edge of the box are red
            for i, expected in zip(range(3), [255, 0, 0]):
                assert all(annotated_array[y1, x1:x2, i] == expected)
                assert all(annotated_array[y2, x1:x2, i] == expected)
                assert all(annotated_array[y1:y2, x1, i] == expected)
                assert all(annotated_array[y1:y2, x2, i] == expected)
            # Make sure almost all the pixels are not changed
            assert ((annotated_array[:, :, 0] == 1).mean()) > threshold
            assert ((annotated_array[:, :, 1] == 1).mean()) > threshold
            assert ((annotated_array[:, :, 2] == 1).mean()) > threshold

    test_image_arr = np.ones((100, 100, 3), dtype="uint8")
    image = Image.fromarray(test_image_arr)
    page = layout.PageLayout(number=1, image=image)
    coords1 = (21, 30, 37, 41)
    rect1 = elements.TextRegion.from_coords(*coords1)
    coords2 = (1, 10, 7, 11)
    rect2 = elements.TextRegion.from_coords(*coords2)
    page.elements = [rect1, rect2]

    annotated_image = page.annotate(colors=colors, add_details=add_details, sources=None)
    check_annotated_image()

    # Scenario 1: where self.image exists
    annotated_image = page.annotate(colors=colors, add_details=add_details)
    check_annotated_image()

    # Scenario 2: where self.image is None, but self.image_path exists
    with patch.object(Image, "open", return_value=image):
        page.image = None
        page.image_path = "mock_path_to_image"
        annotated_image = page.annotate(colors=colors, add_details=add_details)
        check_annotated_image()


@pytest.mark.parametrize(("text", "expected"), [("asdf", "asdf"), (None, "")])
def test_embedded_text_region(text, expected):
    etr = elements.EmbeddedTextRegion.from_coords(0, 0, 24, 24, text=text)
    assert etr.extract_text(objects=None) == expected


class MockDetectionModel(layout.UnstructuredObjectDetectionModel):
    def initialize(self, *args, **kwargs):
        pass

    def predict(self, x):
        return [
            layout.LayoutElement.from_coords(x1=447.0, y1=315.0, x2=1275.7, y2=413.0, text="0"),
            layout.LayoutElement.from_coords(x1=380.6, y1=473.4, x2=1334.8, y2=533.9, text="1"),
            layout.LayoutElement.from_coords(x1=578.6, y1=556.8, x2=1109.0, y2=874.4, text="2"),
            layout.LayoutElement.from_coords(x1=444.5, y1=942.3, x2=1261.1, y2=1584.1, text="3"),
            layout.LayoutElement.from_coords(x1=444.8, y1=1609.4, x2=1257.2, y2=1665.2, text="4"),
            layout.LayoutElement.from_coords(x1=414.0, y1=1718.8, x2=635.0, y2=1755.2, text="5"),
            layout.LayoutElement.from_coords(x1=372.6, y1=1786.9, x2=1333.6, y2=1848.7, text="6"),
        ]


def test_layout_order(mock_image):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_image_path = os.path.join(tmpdir, "mock.jpg")
        mock_image.save(mock_image_path)
        with patch.object(layout, "get_model", lambda: MockDetectionModel()), patch.object(
            layout,
            "convert_pdf_to_image",
            lambda *args, **kwargs: ([mock_image_path]),
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


def test_get_elements_using_image_extraction_raises_with_no_extraction_model(
    mock_image,
):
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
            image_path=None,
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
            fixed_layouts=None,
            pdf_image_dpi=200,
        )


@pytest.mark.parametrize(("pdf_image_dpi", "expected"), [(200, 2200), (100, 1100)])
def test_exposed_pdf_image_dpi(pdf_image_dpi, expected, monkeypatch):
    with patch.object(layout.PageLayout, "from_image") as mock_from_image:
        layout.DocumentLayout.from_file("sample-docs/loremipsum.pdf", pdf_image_dpi=pdf_image_dpi)
        assert mock_from_image.call_args[0][0].height == expected


@pytest.mark.parametrize(
    ("filename", "img_num", "should_complete"),
    [
        ("sample-docs/empty-document.pdf", 0, True),
        ("sample-docs/empty-document.pdf", 10, False),
    ],
)
def test_get_image(filename, img_num, should_complete):
    doc = layout.DocumentLayout.from_file(filename)
    page = doc.pages[0]
    try:
        img = page._get_image(filename, img_num)
        # transform img to numpy array
        img = np.array(img)
        # is a blank image with all pixels white
        assert img.mean() == 255.0
    except ValueError:
        assert not should_complete
