import pytest
from unittest.mock import patch

import unstructured_inference.models.detectron2 as detectron2
import unstructured_inference.models.base as models


class MockDetectron2LayoutModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def detect(self, x):
        return MockLayout()


class MockLayout:
    pass


def test_load_default_model(monkeypatch):
    monkeypatch.setattr(detectron2, "Detectron2LayoutModel", MockDetectron2LayoutModel)

    with patch.object(detectron2, "is_detectron2_available", return_value=True):
        model = models.get_model()

    assert isinstance(model.model, MockDetectron2LayoutModel)


def test_load_default_model_raises_when_not_available():
    with patch.object(detectron2, "is_detectron2_available", return_value=False):
        with pytest.raises(ImportError):
            models.get_model()


@pytest.mark.parametrize("config_path, model_path", [("asdf", "diufs"), ("dfaw", "hfhfhfh")])
def test_load_model(monkeypatch, config_path, model_path):
    monkeypatch.setattr(detectron2, "Detectron2LayoutModel", MockDetectron2LayoutModel)
    with patch.object(detectron2, "is_detectron2_available", return_value=True):
        model = detectron2.load_model(config_path=config_path, model_path=model_path)
    assert config_path == model.model.args[0]
    assert model_path == model.model.kwargs["model_path"]


def test_unstructured_detectron_model():
    model = detectron2.UnstructuredDetectronModel(MockDetectron2LayoutModel())
    assert isinstance(model(None), MockLayout)
