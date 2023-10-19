import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.models.table_transformer.modeling_table_transformer import (
    TableTransformerDecoder,
)

import unstructured_inference.models.table_postprocess as postprocess
from unstructured_inference.models import tables

skip_outside_ci = os.getenv("CI", "").lower() in {"", "false", "f", "0"}


@pytest.fixture()
def table_transformer():
    table_model = tables.UnstructuredTableTransformerModel()
    table_model.initialize(model="microsoft/table-transformer-structure-recognition")
    return table_model


@pytest.fixture()
def example_image():
    return Image.open("./sample-docs/table-multi-row-column-cells.png").convert("RGB")


@pytest.mark.parametrize(
    "model_path",
    [
        ("invalid_table_path"),
        ("incorrect_table_path"),
    ],
)
def test_load_table_model_raises_when_not_available(model_path):
    with pytest.raises(ImportError):
        table_model = tables.UnstructuredTableTransformerModel()
        table_model.initialize(model=model_path)


@pytest.mark.parametrize(
    "model_path",
    [
        "microsoft/table-transformer-structure-recognition",
    ],
)
def test_load_donut_model(model_path):
    table_model = tables.UnstructuredTableTransformerModel()
    table_model.initialize(model=model_path)
    assert type(table_model.model.model.decoder) is TableTransformerDecoder


@pytest.mark.parametrize(
    ("input_test", "output_test"),
    [
        (
            [
                {
                    "label": "table column header",
                    "score": 0.9349299073219299,
                    "bbox": [
                        47.83147430419922,
                        116.8877944946289,
                        2557.79296875,
                        216.98883056640625,
                    ],
                },
                {
                    "label": "table column header",
                    "score": 0.934,
                    "bbox": [
                        47.83147430419922,
                        116.8877944946289,
                        2557.79296875,
                        216.98883056640625,
                    ],
                },
            ],
            [
                {
                    "label": "table column header",
                    "score": 0.9349299073219299,
                    "bbox": [
                        47.83147430419922,
                        116.8877944946289,
                        2557.79296875,
                        216.98883056640625,
                    ],
                },
            ],
        ),
        ([], []),
    ],
)
def test_nms(input_test, output_test):
    output = postprocess.nms(input_test)

    assert output == output_test


@pytest.mark.parametrize(
    ("supercell1", "supercell2"),
    [
        (
            {
                "label": "table spanning cell",
                "score": 0.526617169380188,
                "bbox": [
                    1446.2801513671875,
                    1023.817138671875,
                    2114.3525390625,
                    1099.20166015625,
                ],
                "projected row header": False,
                "header": False,
                "row_numbers": [3, 4],
                "column_numbers": [0, 4],
            },
            {
                "label": "table spanning cell",
                "score": 0.5199193954467773,
                "bbox": [
                    98.92312622070312,
                    676.1566772460938,
                    751.0982666015625,
                    938.5986938476562,
                ],
                "projected row header": False,
                "header": False,
                "row_numbers": [3, 4, 6],
                "column_numbers": [0, 4],
            },
        ),
        (
            {
                "label": "table spanning cell",
                "score": 0.526617169380188,
                "bbox": [
                    1446.2801513671875,
                    1023.817138671875,
                    2114.3525390625,
                    1099.20166015625,
                ],
                "projected row header": False,
                "header": False,
                "row_numbers": [3, 4],
                "column_numbers": [0, 4],
            },
            {
                "label": "table spanning cell",
                "score": 0.5199193954467773,
                "bbox": [
                    98.92312622070312,
                    676.1566772460938,
                    751.0982666015625,
                    938.5986938476562,
                ],
                "projected row header": False,
                "header": False,
                "row_numbers": [4],
                "column_numbers": [0, 4, 6],
            },
        ),
        (
            {
                "label": "table spanning cell",
                "score": 0.526617169380188,
                "bbox": [
                    1446.2801513671875,
                    1023.817138671875,
                    2114.3525390625,
                    1099.20166015625,
                ],
                "projected row header": False,
                "header": False,
                "row_numbers": [3, 4],
                "column_numbers": [1, 4],
            },
            {
                "label": "table spanning cell",
                "score": 0.5199193954467773,
                "bbox": [
                    98.92312622070312,
                    676.1566772460938,
                    751.0982666015625,
                    938.5986938476562,
                ],
                "projected row header": False,
                "header": False,
                "row_numbers": [4],
                "column_numbers": [0, 4, 6],
            },
        ),
        (
            {
                "label": "table spanning cell",
                "score": 0.526617169380188,
                "bbox": [
                    1446.2801513671875,
                    1023.817138671875,
                    2114.3525390625,
                    1099.20166015625,
                ],
                "projected row header": False,
                "header": False,
                "row_numbers": [3, 4],
                "column_numbers": [1, 4],
            },
            {
                "label": "table spanning cell",
                "score": 0.5199193954467773,
                "bbox": [
                    98.92312622070312,
                    676.1566772460938,
                    751.0982666015625,
                    938.5986938476562,
                ],
                "projected row header": False,
                "header": False,
                "row_numbers": [2, 4, 5, 6, 7, 8],
                "column_numbers": [0, 4, 6],
            },
        ),
    ],
)
def test_remove_supercell_overlap(supercell1, supercell2):
    assert postprocess.remove_supercell_overlap(supercell1, supercell2) is None


@pytest.mark.parametrize(
    ("supercells", "rows", "columns", "output_test"),
    [
        (
            [
                {
                    "label": "table spanning cell",
                    "score": 0.9,
                    "bbox": [
                        98.92312622070312,
                        143.11549377441406,
                        2115.197265625,
                        1238.27587890625,
                    ],
                    "projected row header": True,
                    "header": True,
                    "span": True,
                },
            ],
            [
                {
                    "label": "table row",
                    "score": 0.9299452900886536,
                    "bbox": [0, 0, 10, 10],
                    "column header": True,
                    "header": True,
                },
                {
                    "label": "table row",
                    "score": 0.9299452900886536,
                    "bbox": [
                        98.92312622070312,
                        143.11549377441406,
                        2114.3525390625,
                        193.67681884765625,
                    ],
                    "column header": True,
                    "header": True,
                },
                {
                    "label": "table row",
                    "score": 0.9299452900886536,
                    "bbox": [
                        98.92312622070312,
                        143.11549377441406,
                        2114.3525390625,
                        193.67681884765625,
                    ],
                    "column header": True,
                    "header": True,
                },
            ],
            [
                {
                    "label": "table column",
                    "score": 0.9996132254600525,
                    "bbox": [
                        98.92312622070312,
                        143.11549377441406,
                        517.6508178710938,
                        1616.48779296875,
                    ],
                },
                {
                    "label": "table column",
                    "score": 0.9935646653175354,
                    "bbox": [
                        520.0474853515625,
                        143.11549377441406,
                        751.0982666015625,
                        1616.48779296875,
                    ],
                },
            ],
            [
                {
                    "label": "table spanning cell",
                    "score": 0.9,
                    "bbox": [
                        98.92312622070312,
                        143.11549377441406,
                        751.0982666015625,
                        193.67681884765625,
                    ],
                    "projected row header": True,
                    "header": True,
                    "span": True,
                    "row_numbers": [1, 2],
                    "column_numbers": [0, 1],
                },
                {
                    "row_numbers": [0],
                    "column_numbers": [0, 1],
                    "score": 0.9,
                    "propagated": True,
                    "bbox": [
                        98.92312622070312,
                        143.11549377441406,
                        751.0982666015625,
                        193.67681884765625,
                    ],
                },
            ],
        ),
    ],
)
def test_align_supercells(supercells, rows, columns, output_test):
    assert postprocess.align_supercells(supercells, rows, columns) == output_test


@pytest.mark.parametrize(("rows", "bbox", "output"), [([1.0], [0.0], [1.0])])
def test_align_rows(rows, bbox, output):
    assert postprocess.align_rows(rows, bbox) == output


def test_table_prediction_tesseract(table_transformer, example_image):
    prediction = table_transformer.predict(example_image)
    # assert rows spans two rows are detected
    assert '<table><thead><th rowspan="2">' in prediction
    # one of the safest rows to detect should be present
    assert (
        "<tr>"
        "<td>Blind</td>"
        "<td>5</td>"
        "<td>1</td>"
        "<td>4</td>"
        "<td>34.5%, n=1</td>"
        "<td>1199 sec, n=1</td>"
        "</tr>"
    ) in prediction


@pytest.mark.parametrize(
    ("output_format", "expectation"),
    [
        ("html", "<tr><td>Blind</td><td>5</td><td>1</td><td>4</td><td>34.5%, n=1</td>"),
        (
            "cells",
            {
                "column_nums": [0],
                "row_nums": [2],
                "column header": False,
                "cell text": "Blind",
            },
        ),
        ("dataframe", ["Blind", "5", "1", "4", "34.5%, n=1", "1199 sec, n=1"]),
        (None, "<tr><td>Blind</td><td>5</td><td>1</td><td>4</td><td>34.5%, n=1</td>"),
    ],
)
def test_table_prediction_output_format(
    output_format,
    expectation,
    table_transformer,
    example_image,
):
    if output_format:
        result = table_transformer.run_prediction(example_image, result_format=output_format)
    else:
        result = table_transformer.run_prediction(example_image)

    if output_format == "dataframe":
        assert expectation in result.values
    elif output_format == "cells":
        # other output like bbox are flakey to test since they depend on OCR and it may change
        # slightly when OCR pacakge changes or even on different machines
        validation_fields = ("column_nums", "row_nums", "column header", "cell text")
        assert expectation in [{key: cell[key] for key in validation_fields} for cell in result]
    else:
        assert expectation in result


def test_table_prediction_tesseract_with_ocr_tokens(table_transformer, example_image):
    ocr_tokens = [
        {
            # bounding box should match table structure
            "bbox": [70.0, 245.0, 127.0, 266.0],
            "block_num": 0,
            "line_num": 0,
            "span_num": 0,
            "text": "Blind",
        },
    ]
    prediction = table_transformer.predict(example_image, ocr_tokens=ocr_tokens)
    assert prediction == "<table><tr><td>Blind</td></tr></table>"


@pytest.mark.skipif(skip_outside_ci, reason="Skipping paddle test run outside of CI")
def test_table_prediction_paddle(monkeypatch, example_image):
    monkeypatch.setenv("TABLE_OCR", "paddle")
    table_model = tables.UnstructuredTableTransformerModel()

    table_model.initialize(model="microsoft/table-transformer-structure-recognition")
    prediction = table_model.predict(example_image)
    # Note(yuming): lossen paddle table prediction output test since performance issue
    # and results are different in different platforms (i.e., gpu vs cpu)
    assert len(prediction)


def test_table_prediction_invalid_table_ocr(monkeypatch, example_image):
    monkeypatch.setenv("TABLE_OCR", "invalid_table_ocr")
    with pytest.raises(ValueError):
        table_model = tables.UnstructuredTableTransformerModel()

        table_model.initialize(model="microsoft/table-transformer-structure-recognition")
        _ = table_model.predict(example_image)


def test_intersect():
    a = postprocess.Rect()
    b = postprocess.Rect([1, 2, 3, 4])
    assert a.intersect(b).get_area() == 4.0


def test_include_rect():
    a = postprocess.Rect()
    assert a.include_rect([1, 2, 3, 4]).get_area() == 4.0


@pytest.mark.parametrize(
    ("spans", "join_with_space", "expected"),
    [
        (
            [
                {
                    "flags": 2**0,
                    "text": "5",
                    "superscript": False,
                    "span_num": 0,
                    "line_num": 0,
                    "block_num": 0,
                },
            ],
            True,
            "",
        ),
        (
            [
                {
                    "flags": 2**0,
                    "text": "p",
                    "superscript": False,
                    "span_num": 0,
                    "line_num": 0,
                    "block_num": 0,
                },
            ],
            True,
            "p",
        ),
        (
            [
                {
                    "flags": 2**0,
                    "text": "p",
                    "superscript": False,
                    "span_num": 0,
                    "line_num": 0,
                    "block_num": 0,
                },
                {
                    "flags": 2**0,
                    "text": "p",
                    "superscript": False,
                    "span_num": 0,
                    "line_num": 0,
                    "block_num": 0,
                },
            ],
            True,
            "p p",
        ),
        (
            [
                {
                    "flags": 2**0,
                    "text": "p",
                    "superscript": False,
                    "span_num": 0,
                    "line_num": 0,
                    "block_num": 0,
                },
                {
                    "flags": 2**0,
                    "text": "p",
                    "superscript": False,
                    "span_num": 0,
                    "line_num": 0,
                    "block_num": 1,
                },
            ],
            True,
            "p p",
        ),
        (
            [
                {
                    "flags": 2**0,
                    "text": "p",
                    "superscript": False,
                    "span_num": 0,
                    "line_num": 0,
                    "block_num": 0,
                },
                {
                    "flags": 2**0,
                    "text": "p",
                    "superscript": False,
                    "span_num": 0,
                    "line_num": 0,
                    "block_num": 1,
                },
            ],
            False,
            "p p",
        ),
    ],
)
def test_extract_text_from_spans(spans, join_with_space, expected):
    res = postprocess.extract_text_from_spans(
        spans,
        join_with_space=join_with_space,
        remove_integer_superscripts=True,
    )
    assert res == expected


@pytest.mark.parametrize(
    ("supercells", "expected_len"),
    [
        ([{"header": "hi", "row_numbers": [0, 1, 2], "score": 0.9}], 1),
        (
            [
                {
                    "header": "hi",
                    "row_numbers": [0],
                    "column_numbers": [1, 2, 3],
                    "score": 0.9,
                },
                {
                    "header": "hi",
                    "row_numbers": [1],
                    "column_numbers": [1],
                    "score": 0.9,
                },
                {
                    "header": "hi",
                    "row_numbers": [1],
                    "column_numbers": [2],
                    "score": 0.9,
                },
                {
                    "header": "hi",
                    "row_numbers": [1],
                    "column_numbers": [3],
                    "score": 0.9,
                },
            ],
            4,
        ),
        (
            [
                {
                    "header": "hi",
                    "row_numbers": [0],
                    "column_numbers": [0],
                    "score": 0.9,
                },
                {
                    "header": "hi",
                    "row_numbers": [1],
                    "column_numbers": [0],
                    "score": 0.9,
                },
                {
                    "header": "hi",
                    "row_numbers": [1, 2],
                    "column_numbers": [0],
                    "score": 0.9,
                },
                {
                    "header": "hi",
                    "row_numbers": [3],
                    "column_numbers": [0],
                    "score": 0.9,
                },
            ],
            3,
        ),
    ],
)
def test_header_supercell_tree(supercells, expected_len):
    postprocess.header_supercell_tree(supercells)
    assert len(supercells) == expected_len


def test_cells_to_html():
    # example table
    # +----------+---------------------+
    # |    two   |   two columns       |
    # |          |----------+----------|
    # |    rows  |sub cell 1|sub cell 2|
    # +----------+----------+----------+
    cells = [
        {"row_nums": [0, 1], "column_nums": [0], "cell text": "two row", "column header": False},
        {"row_nums": [0], "column_nums": [1, 2], "cell text": "two cols", "column header": False},
        {"row_nums": [1], "column_nums": [1], "cell text": "sub cell 1", "column header": False},
        {"row_nums": [1], "column_nums": [2], "cell text": "sub cell 2", "column header": False},
    ]
    expected = (
        '<table><tr><td rowspan="2">two row</td><td colspan="2">two '
        "cols</td></tr><tr><td></td><td>sub cell 1</td><td>sub cell 2</td></tr></table>"
    )
    assert tables.cells_to_html(cells) == expected


def test_auto_zoom(mocker):
    spy = mocker.spy(tables, "zoom_image")
    model = tables.UnstructuredTableTransformerModel()
    model.initialize("microsoft/table-transformer-structure-recognition")
    image = Image.open(
        Path(os.path.dirname(os.path.abspath(__file__)))
        / ".."
        / ".."
        / "sample-docs"
        / "layout-parser-paper-fast.jpg",
    )
    model.get_tokens(image)
    assert spy.call_count == 1


@pytest.mark.parametrize("zoom", [1, 0.1, 5, -1, 0])
def test_zoom_image(example_image, zoom):
    width, height = example_image.size
    new_image = tables.zoom_image(example_image, zoom)
    new_w, new_h = new_image.size
    if zoom <= 0:
        zoom = 1
    assert new_w == np.round(width * zoom, 0)
    assert new_h == np.round(height * zoom, 0)


def test_padded_results_has_right_dimensions(table_transformer, example_image):
    str_class_name2idx = tables.get_class_map("structure")
    # a simpler mapping so we keep all structure in the returned objs below for test
    str_class_idx2name = {v: "table cell" for v in str_class_name2idx.values()}
    # pad size is no more than 10% of the original image so we can setup test below easier
    pad = int(min(example_image.size) / 10)

    structure = table_transformer.get_structure(example_image, pad_for_structure_detection=pad)
    # boxes deteced OUTSIDE of the original image; this shouldn't happen but we want to make sure
    # the code handles it as expected
    structure["pred_boxes"][0][0, :2] = 0.5
    structure["pred_boxes"][0][0, 2:] = 1.0
    # mock a box we know are safly inside the original image with known positions
    width, height = example_image.size
    padded_width = width + pad * 2
    padded_height = height + pad * 2
    original = [1, 3, 101, 53]
    structure["pred_boxes"][0][1, :] = torch.tensor(
        [
            (51 + pad) / padded_width,
            (28 + pad) / padded_height,
            100 / padded_width,
            50 / padded_height,
        ],
    )
    objs = tables.outputs_to_objects(structure, example_image.size, str_class_idx2name)
    np.testing.assert_almost_equal(objs[0]["bbox"], [-pad, -pad, width + pad, height + pad], 4)
    np.testing.assert_almost_equal(objs[1]["bbox"], original, 4)
    # a more strict test would be to constrain the actual detected boxes to be within the original
    # image but that requires the table transformer to behave in certain ways and do not
    # actually test the padding math; so here we use the relaxed condition
    for obj in objs[2:]:
        x1, y1, x2, y2 = obj["bbox"]
        assert max(x1, x2) < width + pad
        assert max(y1, y2) < height + pad
