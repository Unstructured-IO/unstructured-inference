from unittest import mock

import numpy as np
import torch
from PIL import Image

from unstructured_inference.constants import Source
from unstructured_inference.models import super_gradients


def test_supergradients_model():
    model_path = "/home/ec2-user/downloaded_s3/average_model.onnx"
    model = super_gradients.UnstructuredSuperGradients()
    model.initialize(
        model_path=model_path,
        label_map={0: 'Title', 1: 'Text', 2: 'Footer', 4: 'Picture', 5:'ListItem', 6:'Caption', 7:'Header'},
        input_shape=(640,640),
    )
    img = Image.open("sample-docs/loremipsum.jpg")
    el, *_ = model(img)
    assert el.source == Source.SUPER_GRADIENTS
    assert el.prob == 0.7743491
    assert el.type == 'Title'
