from unittest import mock

import pytest
import torch
from PIL import Image

from unstructured_inference.models import chipper
from unstructured_inference.models.base import get_model


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
        model.initialize("", "", "", "", "", "", "", "", "")
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
    self.model.encoder = mock.MagicMock()
    self.stopping_criteria = mock.MagicMock()
    self.processor = mock.MagicMock()
    self.logits_processor = mock.MagicMock()
    self.input_ids = mock.MagicMock()
    self.device = "cpu"
    self.max_length = 1200


def test_predict_tokens():
    with mock.patch.object(chipper.UnstructuredChipperModel, "initialize", mock_initialize):
        model = chipper.UnstructuredChipperModel()
        model.initialize()
        with open("sample-docs/loremipsum.png", "rb") as fp:
            im = Image.open(fp)
            tokens, _ = model.predict_tokens(im)
        assert tokens == [5, 4, 3, 2, 1]


@pytest.mark.parametrize(
    ("decoded_str", "expected_classes", "expected_parent_ids"),
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
        swap_head_hidden_layer_size=128,
        start_token_prefix="<s_",
        max_length=1200,
        heatmap_h=40,
        heatmap_w=30,
        source="chipper",
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


def test_no_repeat_ngram_logits():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4]])
    logits = torch.tensor([[0.1, -0.3, -0.5, 0, 1.0, -0.9]])

    batch_size = input_ids.shape[0]
    cur_len = input_ids.shape[-1]

    no_repeat_ngram_size = 2

    logitsProcessor = chipper.NoRepeatNGramLogitsProcessor(ngram_size=2, context_length=10)
    output = logitsProcessor(input_ids=input_ids, scores=logits)

    assert (
        int(
            torch.sum(
                output == torch.tensor([[0.1000, -0.3000, -0.5000, 0.0000, 1.0000, -float("inf")]]),
            ),
        )
        == 6
    )

    skip_tokens = {5}

    output = chipper._no_repeat_ngram_logits(
        input_ids=input_ids,
        cur_len=cur_len,
        logits=logits,
        batch_size=batch_size,
        no_repeat_ngram_size=no_repeat_ngram_size,
        skip_tokens=skip_tokens,
    )

    assert (
        int(
            torch.sum(
                output == logits,
            ),
        )
        == 6
    )

    output = chipper._no_repeat_ngram_logits(
        input_ids=input_ids,
        cur_len=cur_len,
        logits=logits,
        batch_size=batch_size,
        no_repeat_ngram_size=12,
    )

    assert (
        int(
            torch.sum(
                output == logits,
            ),
        )
        == 6
    )


def test_ngram_repetiton_stopping_criteria():
    input_ids = torch.tensor([[1, 2, 3, 4, 0, 1, 2, 3, 4]])
    logits = torch.tensor([[0.1, -0.3, -0.5, 0, 1.0, -0.9]])

    stoppingCriteria = chipper.NGramRepetitonStoppingCriteria(
        ngram_size=2, context_length=10, skip_tokens={0, 1, 2, 3, 4}
    )

    output = stoppingCriteria(input_ids=input_ids, scores=logits)

    assert output is False

    stoppingCriteria = chipper.NGramRepetitonStoppingCriteria(
        ngram_size=2, context_length=10, skip_tokens={1, 2, 3, 4}
    )
    output = stoppingCriteria(input_ids=input_ids, scores=logits)
    assert output is True


def test_no_repeat_group_ngram_logits_processor():
    input_ids = torch.tensor([[1, 2, 3, 4, 0, 1, 2, 3, 4]])
    logits = torch.tensor([[0.1, -0.3, -0.5, 0, 1.0, -0.9]])

    logitsProcessor = chipper.NoRepeatGroupNGramLogitsProcessor(ngram_size=3, token_group=[1, 2])

    output = logitsProcessor(input_ids=input_ids, scores=logits)

    assert (
        int(
            torch.sum(
                output == torch.tensor([[0.1000, -0.3000, -0.5000, 0.0000, 1.0000, -0.9000]]),
            ),
        )
        == 6
    )


def test_target_token_id_stopping_criterion():
    input_ids = torch.tensor([1, 2, 3])
    logits = torch.tensor([0.1, 0.2, 0.3])

    stoppingCriterion = chipper.TargetTokenIdStoppingCriterion(1)

    output = stoppingCriterion(input_ids=input_ids, scores=logits)

    assert output is True


@pytest.mark.parametrize(
    ("decoded_str", "expected_classes"),
    [
        (
            "<s><s_Misc> 1</s_Misc><s_Text>There is some text here.</s_Text></s>",
            ["Misc", "Text"],
        ),
        (
            "<s><s_List><s_List-item>Text here.</s_List-item><s_List><s_List-item>Final one",
            ["List", "List-item", "List", "List-item"],
        ),
    ],
)
def test_postprocess_bbox(decoded_str, expected_classes):
    model = chipper.UnstructuredChipperModel()
    pre_trained_model = "unstructuredio/ved-fine-tuning"
    model.initialize(
        pre_trained_model,
        prompt="<s>",
        swap_head=False,
        swap_head_hidden_layer_size=128,
        start_token_prefix="<s_",
        max_length=1200,
        heatmap_h=40,
        heatmap_w=30,
        source="chipper",
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

    assert len(out) == len(expected_classes)

    for i in range(len(out)):
        assert out[i].bbox is not None
        assert out[i].type == expected_classes[i]


def test_predict_tokens_beam_indices():
    model = get_model("chipper")
    model.stopping_criteria = [
        chipper.NGramRepetitonStoppingCriteria(
            ngram_size=1,
            context_length=10,
            skip_tokens={},
        ),
    ]
    img = Image.open("sample-docs/easy_table.jpg")
    output = model.predict_tokens(image=img)
    assert len(output) > 0


def test_largest_margin_edge():
    model = get_model("chipper")
    img = Image.open("sample-docs/easy_table.jpg")
    output = model.largest_margin(image=img, input_bbox=[0, 1, 0, 0], transpose=False)

    assert output is None

    output = model.largest_margin(img, [1, 1, 1, 1], False)

    assert output is None

    output = model.largest_margin(img, [2, 1, 3, 10], True)

    assert output == (0, 0, 0)


def test_deduplicate_detected_elements():
    model = get_model("chipper")
    img = Image.open("sample-docs/easy_table.jpg")
    elements = model(img)

    output = model.deduplicate_detected_elements(elements)

    assert len(output) == 2


def test_logitsprocessor_exception():
    with pytest.raises(ValueError):
        chipper.NoRepeatNGramLogitsProcessor(ngram_size="", context_length=10)

    with pytest.raises(ValueError):
        chipper.NoRepeatGroupNGramLogitsProcessor(ngram_size="", token_group={})


def test_run_chipper_v3():
    model = get_model("chipper")
    img = Image.open("sample-docs/easy_table.jpg")
    elements = model(img)
    tables = [el for el in elements if el.type == "Table"]
    assert all(table.text_as_html.startswith("<table>") for table in tables)
    assert all("<table>" not in table.text for table in tables)


@pytest.mark.parametrize(
    ("bbox", "output"),
    [
        (
            [0, 0, 0, 0],
            None,
        ),
        (
            [0, 1, 1, -1],
            None,
        ),
    ],
)
def test_largest_margin(bbox, output):
    model = get_model("chipper")
    img = Image.open("sample-docs/easy_table.jpg")
    assert model.largest_margin(img, bbox) is output


@pytest.mark.parametrize(
    ("bbox", "output"),
    [
        (
            [0, 1, 0, -1],
            [0, 1, 0, -1],
        ),
        (
            [0, 1, 1, -1],
            [0, 1, 1, -1],
        ),
        (
            [20, 10, 30, 40],
            [20, 10, 30, 40],
        ),
    ],
)
def test_reduce_bbox_overlap(bbox, output):
    model = get_model("chipper")
    img = Image.open("sample-docs/easy_table.jpg")
    assert model.reduce_bbox_overlap(img, bbox) == output


@pytest.mark.parametrize(
    ("bbox", "output"),
    [
        (
            [20, 10, 30, 40],
            [20, 10, 30, 40],
        ),
    ],
)
def test_reduce_bbox_no_overlap(bbox, output):
    model = get_model("chipper")
    img = Image.open("sample-docs/easy_table.jpg")
    assert model.reduce_bbox_no_overlap(img, bbox) == output


@pytest.mark.parametrize(
    ("bbox1", "bbox2", "output"),
    [
        (
            [0, 50, 20, 80],
            [10, 10, 30, 30],
            (
                "horizontal",
                [10, 10, 30, 30],
                [0, 50, 20, 80],
                [0, 50, 20, 80],
                [10, 10, 30, 30],
                None,
            ),
        ),
        (
            [10, 10, 30, 30],
            [40, 10, 60, 30],
            (
                "vertical",
                [40, 10, 60, 30],
                [10, 10, 30, 30],
                [10, 10, 30, 30],
                [40, 10, 60, 30],
                None,
            ),
        ),
        (
            [10, 80, 30, 100],
            [40, 10, 60, 30],
            (
                "none",
                [40, 10, 60, 30],
                [10, 80, 30, 100],
                [10, 80, 30, 100],
                [40, 10, 60, 30],
                None,
            ),
        ),
        (
            [40, 10, 60, 30],
            [10, 10, 30, 30],
            (
                "vertical",
                [10, 10, 30, 30],
                [40, 10, 60, 30],
                [10, 10, 30, 30],
                [40, 10, 60, 30],
                None,
            ),
        ),
    ],
)
def test_check_overlap(bbox1, bbox2, output):
    model = get_model("chipper")

    assert model.check_overlap(bbox1, bbox2) == output
