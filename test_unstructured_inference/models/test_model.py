from typing import Any
from unittest import mock

import pytest

import unstructured_inference.models.base as models
from unstructured_inference.models.unstructuredmodel import (
    ModelNotInitializedError,
    UnstructuredObjectDetectionModel,
)


class MockModel(UnstructuredObjectDetectionModel):
    call_count = 0

    def __init__(self):
        self.initializer = mock.MagicMock()
        super().__init__()

    def initialize(self, *args, **kwargs):
        return self.initializer(self, *args, **kwargs)

    def predict(self, x: Any) -> Any:
        return []


def test_get_model(monkeypatch):
    monkeypatch.setattr(models, "models", {})
    monkeypatch.setattr(
        models,
        "UnstructuredDetectronModel",
        MockModel,
    )
    assert isinstance(models.get_model("checkbox"), MockModel)


def test_get_model_warns_on_chipper(monkeypatch, caplog):
    monkeypatch.setattr(
        models,
        "UnstructuredChipperModel",
        MockModel,
    )
    with mock.patch.object(models, "models", {}):
        models.get_model("chipper")
        assert caplog.records[0].levelname == "WARNING"


def test_raises_invalid_model():
    with pytest.raises(models.UnknownModelException):
        models.get_model("fake_model")


def test_raises_uninitialized():
    with pytest.raises(ModelNotInitializedError):
        models.UnstructuredDetectronModel().predict(None)


def test_model_initializes_once():
    from unstructured_inference.inference import layout

    with mock.patch.object(models, "UnstructuredYoloXModel", MockModel), mock.patch.object(
        models,
        "models",
        {},
    ):
        doc = layout.DocumentLayout.from_file("sample-docs/loremipsum.pdf")
        doc.pages[0].detection_model.initializer.assert_called_once()
        assert hasattr(
            doc.pages[0].elements[0],
            "prob",
        )  # NOTE(pravin) New Assertion to Make Sure Elements have probability attribute
        assert (
            doc.pages[0].elements[0].prob is None
        )  # NOTE(pravin) New Assertion to Make Sure Uncategorized Text has None Probability


def test_deduplicate_detected_elements():
    import numpy as np

    from unstructured_inference.inference.elements import intersections
    from unstructured_inference.inference.layout import DocumentLayout
    from unstructured_inference.models.base import get_model

    model = get_model("yolox_quantized")
    # model.confidence_threshold=0.5
    file = "sample-docs/example_table.jpg"
    doc = DocumentLayout.from_image_file(
        file,
        model,
        ocr_strategy="never",
        supplement_with_ocr_elements=False,
    )
    known_elements = [e for e in doc.pages[0].elements if e.type != "UncategorizedText"]
    # Compute intersection matrix
    intersections_mtx = intersections(*known_elements)
    # Get rid off diagonal (cause an element will always intersect itself)
    np.fill_diagonal(intersections_mtx, False)
    # Now all the elements should be False, because any intersection remains
    return not intersections_mtx.all()


def test_enhance_regions():
    from unstructured_inference.inference.elements import Rectangle
    from unstructured_inference.models.base import get_model

    elements = [
        Rectangle(0, 0, 1, 1),
        Rectangle(0.01, 0.01, 1.01, 1.01),
        Rectangle(0.02, 0.02, 1.02, 1.02),
        Rectangle(0.03, 0.03, 1.03, 1.03),
        Rectangle(0.04, 0.04, 1.04, 1.04),
        Rectangle(0.05, 0.05, 1.05, 1.05),
        Rectangle(0.06, 0.06, 1.06, 1.06),
        Rectangle(0.07, 0.07, 1.07, 1.07),
        Rectangle(0.08, 0.08, 1.08, 1.08),
        Rectangle(0.09, 0.09, 1.09, 1.09),
        Rectangle(0.10, 0.10, 1.10, 1.10),
    ]
    model = get_model("yolox_tiny")
    elements = model.enhance_regions(elements, 0.5)
    assert len(elements) == 1
    assert (elements[0].x1, elements[0].y1, elements[0].x2, elements[0].x2) == (0, 0, 1.10, 1.10)


def test_clean_tables():
    from unstructured_inference.inference.layout import LayoutElement
    from unstructured_inference.models.base import get_model

    elements = [
        LayoutElement(0, 0, 1, 1, type="Table"),
        LayoutElement(0.01, 0.01, 1.01, 1.01),
        LayoutElement(0.02, 0.02, 1.02, 1.02),
        LayoutElement(0.03, 0.03, 1.03, 1.03),
        LayoutElement(0.04, 0.04, 1.04, 1.04),
        LayoutElement(0.05, 0.05, 1.05, 1.05),
    ]
    model = get_model("yolox_tiny")
    elements = model.enhance_regions(elements, 0.5)
    assert len(elements) == 1
    assert (elements[0].x1, elements[0].y1, elements[0].x2, elements[0].x2) == (0, 0, 1.05, 1.05)
