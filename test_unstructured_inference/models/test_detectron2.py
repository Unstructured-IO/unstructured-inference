import pytest
from unittest.mock import patch

import unstructured_inference.models.detectron2 as detectron2
import unstructured_inference.models.base as models


class MockDetectron2LayoutModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def image_processing(self, x):
        return []


def test_load_default_model():
    with patch.object(detectron2.onnxruntime, "InferenceSession", new=MockDetectron2LayoutModel):
        model = models.get_model()

    assert isinstance(model.model, MockDetectron2LayoutModel)


@pytest.mark.parametrize(("model_path", "label_map"), [("asdf", "diufs"), ("dfaw", "hfhfhfh")])
def test_load_model(model_path, label_map):
    with patch.object(detectron2.onnxruntime, "InferenceSession", return_value=True):
        model = detectron2.UnstructuredDetectronModel()
        model.initialize(model_path=model_path, label_map=label_map)
        args, _ = detectron2.onnxruntime.InferenceSession.call_args
        assert args == (model_path,)
    assert label_map == model.label_map


def test_unstructured_detectron_model():
    model = detectron2.UnstructuredDetectronModel()
    model.model = MockDetectron2LayoutModel()
    with patch.object(detectron2.UnstructuredDetectronModel, "image_processing", return_value=[]):
        result = model(None)
    assert isinstance(result, list)
    assert len(result) == 0
