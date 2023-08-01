from typing import Any
from unittest import mock

import pytest

import unstructured_inference.models.base as models
from unstructured_inference.models.unstructuredmodel import (
    ModelNotInitializedError,
    UnstructuredObjectDetectionModel,
)


class MockModel(UnstructuredObjectDetectionModel):
    initialize = mock.MagicMock()
    # def initialize(self, *args, **kwargs):
    #     pass

    def predict(self, x: Any) -> Any:
        return []


def test_get_model(monkeypatch):
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
    models.get_model("chipper")
    assert caplog.records[0].levelname == "WARNING"


def test_raises_invalid_model():
    with pytest.raises(models.UnknownModelException):
        models.get_model("fake_model")


def test_raises_uninitialized():
    with pytest.raises(ModelNotInitializedError):
        models.UnstructuredDetectronModel().predict(None)


def test_model_initializes_once():
    with mock.patch.object(models, "UnstructuredDetectronONNXModel", MockModel) as f:
        from unstructured_inference.inference.layout import DocumentLayout

        DocumentLayout.from_file("sample-docs/layout-parser-paper.pdf")
        f.initialize.assert_called_once()
