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

    initialize = mock.MagicMock()

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

    with mock.patch.object(models, "UnstructuredDetectronONNXModel", MockModel), mock.patch.object(
        models,
        "models",
        {},
    ):
        doc = layout.DocumentLayout.from_file("sample-docs/layout-parser-paper.pdf")

        doc.pages[0].detection_model.initializer.assert_called_once()
