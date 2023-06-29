from unittest import mock

import pytest
from PIL import Image

from unstructured_inference.models import largemodel


def test_initialize():
    with mock.patch.object(
        largemodel.AutoTokenizer,
        "from_pretrained",
    ) as mock_tokenizer, mock.patch.object(
        largemodel,
        "DonutProcessor",
    ) as mock_donut_processor, mock.patch.object(
        largemodel,
        "DonutImageProcessor",
    ) as mock_donut_image_processor, mock.patch.object(
        largemodel.VisionEncoderDecoderModel,
        "from_pretrained",
    ) as mock_vision_encoder_decoder_model:
        model = largemodel.UnstructuredLargeModel()
        model.initialize("", "", "")
        mock_tokenizer.assert_called_once()
        mock_donut_processor.assert_called_once()
        mock_donut_image_processor.assert_called_once()
        mock_vision_encoder_decoder_model.assert_called_once()


class MockToList:
    def tolist(self):
        return [[5, 4, 3, 2, 1]]


class MockModel:
    def generate(*args, **kwargs):
        return MockToList()


def mock_initialize(self, *arg, **kwargs):
    self.model = MockModel()
    self.processor = mock.MagicMock()


def test_predict_tokens():
    with mock.patch.object(largemodel.UnstructuredLargeModel, "initialize", mock_initialize):
        model = largemodel.UnstructuredLargeModel()
        model.initialize()
        with open("sample-docs/loremipsum.png", "rb") as fp:
            im = Image.open(fp)
            tokens = model.predict_tokens(im)
        assert tokens[1:-1] == [5, 4, 3, 2, 1]


@pytest.mark.parametrize(
    ("decoded_str", "expected_classes"),
    [
        (
            "<s_Title>Hi buddy!</s_Title><s_Text>There is some text here.</s_Text>",
            ["Title", "Text"],
        ),
        ("<s_Title>Hi buddy!</s_Title><s_Text>There is some text here.", ["Title", "Text"]),
    ],
)
def test_postprocess(decoded_str, expected_classes):
    with mock.patch.object(largemodel.UnstructuredLargeModel, "initialize", mock_initialize):
        pass
    model = largemodel.UnstructuredLargeModel()
    tokenizer_model = "xlm-roberta-large"
    pre_trained_model = "nielsr/donut-base"
    model.initialize(tokenizer_model, pre_trained_model, None)

    tokens = model.tokenizer.encode(decoded_str)
    out = model.postprocess(tokens)
    assert len(out) == 2
    element1, element2 = out

    assert [element1.type, element2.type] == expected_classes


def test_predict():
    with mock.patch.object(
        largemodel.UnstructuredLargeModel,
        "predict_tokens",
    ) as mock_predict_tokens, mock.patch.object(
        largemodel.UnstructuredLargeModel,
        "postprocess",
    ) as mock_postprocess:
        model = largemodel.UnstructuredLargeModel()
        model.predict("hello")
        mock_predict_tokens.assert_called_once()
        mock_postprocess.assert_called_once()
