import pytest

import unstructured_inference.models.base as models


class MockModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_get_model(monkeypatch):
    monkeypatch.setattr(models, "load_model", lambda *args, **kwargs: MockModel(*args, **kwargs))
    monkeypatch.setattr(
        models,
        "_get_model_loading_info",
        lambda *args, **kwargs: (
            "fake-binary-path",
            "fake-config-path",
            {0: "Unchecked", 1: "Checked"},
            None,
        ),
    )
    assert isinstance(models.get_model("checkbox"), MockModel)


def test_raises_invalid_model():
    with pytest.raises(models.UnknownModelException):
        models.get_model("fake_model")
