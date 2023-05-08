import os
from unittest.mock import patch

import pytest
from PIL import Image

import unstructured_inference.models.detectron2 as detectron2
import unstructured_inference.models.base as models


class MockDetectron2LayoutModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def run(self, *args):
        return ([(1, 2, 3, 4)], [0], [0.818], [(4, 5)])

    def get_inputs(self):
        class input_thing:
            name = "Bernard"

        return [input_thing()]


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
    model.model = 1
    with patch.object(detectron2.UnstructuredDetectronModel, "predict", return_value=[]):
        result = model(None)
    assert isinstance(result, list)
    assert len(result) == 0


def test_inference():
    with patch.object(
        detectron2.onnxruntime, "InferenceSession", return_value=MockDetectron2LayoutModel()
    ):
        model = detectron2.UnstructuredDetectronModel()
        model.initialize(model_path="test_path", label_map={0: "test_class"})
        assert isinstance(model.model, MockDetectron2LayoutModel)
        with open(os.path.join("sample-docs", "receipt-sample.jpg"), mode="rb") as fp:
            image = Image.open(fp)
            image.load()
        elements = model(image)
        assert len(elements) == 1
        element = elements[0]
        (x1, y1), _, (x2, y2), _ = element.coordinates
        assert x2 / x1 == pytest.approx(3.0)
        assert y2 / y1 == pytest.approx(2.0)
        assert element.type == "test_class"
