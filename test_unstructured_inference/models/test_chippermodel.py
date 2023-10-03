from unittest import mock

import pytest
import torch
from PIL import Image

from unstructured_inference.models import chipper


def test_initialize():
    with mock.patch.object(
        chipper.DonutProcessor,
        "from_pretrained",
    ) as mock_donut_processor, mock.patch.object(
        chipper,
        "NoRepeatNGramLogitsProcessor",
    ) as mock_logits_processor, mock.patch.object(
        chipper.VisionEncoderDecoderModel,
        "from_pretrained",
    ) as mock_vision_encoder_decoder_model:
        model = chipper.UnstructuredChipperModel()
        model.initialize("", "", "", "", "", "")
        mock_donut_processor.assert_called_once()
        mock_logits_processor.assert_called_once()
        mock_vision_encoder_decoder_model.assert_called_once()


class MockToList:
    def tolist(self):
        return [[5, 4, 3, 2, 1]]


class MockModel:
    def generate(*args, **kwargs):
        return {"cross_attentions": mock.MagicMock(), "sequences": [[5, 4, 3, 2, 1]]}
        return MockToList()


def mock_initialize(self, *arg, **kwargs):
    self.model = MockModel()
    self.processor = mock.MagicMock()
    self.logits_processor = mock.MagicMock()
    self.input_ids = mock.MagicMock()
    self.device = "cpu"


def test_predict_tokens():
    with mock.patch.object(chipper.UnstructuredChipperModel, "initialize", mock_initialize):
        model = chipper.UnstructuredChipperModel()
        model.initialize()
        with open("sample-docs/loremipsum.png", "rb") as fp:
            im = Image.open(fp)
            tokens, _ = model.predict_tokens(im)
        assert tokens == [5, 4, 3, 2, 1]


@pytest.mark.parametrize(
    ("decoded_str", "expected_classes", "expected_ids", "expected_parent_ids"),
    [
        (
            "<s_Title>Hi buddy!</s_Title><s_Text>There is some text here.</s_Text>",
            ["Title", "Text"],
            [None, None],
        ),
        (
            "<s_Title>Hi buddy!</s_Title><s_Text>There is some text here.",
            ["Title", "Text"],
            [None, None],
        ),
        (
            "<s_List><s_List-item>Hi buddy!</s_List-item></s_List>",
            ["List", "List-item"],
            [None, 0],
        ),
    ],
)
def test_postprocess(decoded_str, expected_classes, expected_parent_ids):
    model = chipper.UnstructuredChipperModel()
    pre_trained_model = "unstructuredio/ved-fine-tuning"
    model.initialize(
        pre_trained_model,
        prompt="<s>",
        swap_head=False,
        max_length=1200,
        heatmap_h=40,
        heatmap_w=30,
    )

    tokens = model.tokenizer.encode(decoded_str)
    cross_attentions = [
        [torch.ones([1, 16, 2, 1200]) if j == 0 else torch.ones([1, 16, 1, 1200]) for i in range(4)]
        for j in range(550)
    ]
    with open("sample-docs/loremipsum.png", "rb") as fp:
        out = model.postprocess(
            image=Image.open(fp),
            output_ids=tokens,
            decoder_cross_attentions=cross_attentions,
        )
    assert len(out) == 2
    element1, element2 = out

    assert [element1.type, element2.type] == expected_classes

    assert [element1.parent, element2.parent] == [
        element1 if expected_parent_id is not None else None
        for expected_parent_id in expected_parent_ids
    ]


def test_predict():
    with mock.patch.object(
        chipper.UnstructuredChipperModel,
        "predict_tokens",
        mock.MagicMock(return_value=(mock.MagicMock(), mock.MagicMock())),
    ) as mock_predict_tokens, mock.patch.object(
        chipper.UnstructuredChipperModel,
        "postprocess",
    ) as mock_postprocess:
        model = chipper.UnstructuredChipperModel()
        model.predict("hello")
        mock_predict_tokens.assert_called_once()
        mock_postprocess.assert_called_once()
