import pytest

from unstructured_inference import models


class MockModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_get_model(monkeypatch):
    monkeypatch.setattr(models, "load_model", lambda *args, **kwargs: MockModel(*args, **kwargs))
    assert isinstance(models.get_model("checkbox"), MockModel)


def test_raises_invalid_model():
    with pytest.raises(ValueError):
        models.get_model("fake_model")
