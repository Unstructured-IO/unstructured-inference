import pytest
from unittest.mock import patch

import unstructured_inference.models.detectron2 as detectron2


class MockDetectron2LayoutModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_load_default_model(monkeypatch):
    monkeypatch.setattr(detectron2, "Detectron2LayoutModel", MockDetectron2LayoutModel)

    with patch.object(detectron2, "is_detectron2_available", return_value=True):
        model = detectron2.load_default_model()

    assert isinstance(model, MockDetectron2LayoutModel)


def test_load_default_model_raises_when_not_available():
    with patch.object(detectron2, "is_detectron2_available", return_value=False):
        with pytest.raises(ImportError):
            detectron2.load_default_model()


@pytest.mark.parametrize("config_path, model_path", [("asdf", "diufs"), ("dfaw", "hfhfhfh")])
def test_load_model(monkeypatch, config_path, model_path):
    monkeypatch.setattr(detectron2, "Detectron2LayoutModel", MockDetectron2LayoutModel)
    with patch.object(detectron2, "is_detectron2_available", return_value=True):
        model = detectron2.load_model(config_path=config_path, model_path=model_path)
    assert config_path == model.args[0]
    assert model_path == model.kwargs["model_path"]
