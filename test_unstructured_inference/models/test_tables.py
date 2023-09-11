from unittest.mock import patch

import pytest
from transformers.models.table_transformer.modeling_table_transformer import (
    TableTransformerDecoder,
)

import unstructured_inference.models.table_postprocess as postprocess
from unstructured_inference.models import tables


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


# TODO: break this test down so it doesn't account for nearly 8% of test coverage
@pytest.mark.parametrize(
    ("model_path", "table_ocr"),
    [
        ("microsoft/table-transformer-structure-recognition", "paddle"),
        ("microsoft/table-transformer-structure-recognition", "tesseract"),
    ],
)
def test_table_prediction(model_path, table_ocr, monkeypatch):
    monkeypatch.setenv("TABLE_OCR", table_ocr)
    table_model = tables.UnstructuredTableTransformerModel()
    from PIL import Image

    table_model.initialize(model=model_path)
    img = Image.open("./sample-docs/table-multi-row-column-cells.png").convert("RGB")
    prediction = table_model.predict(img)
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
    
def test_table_prediction_invalid_table_ocr(model_path, monkeypatch):
    monkeypatch.setenv("TABLE_OCR", "invalid_table_ocr")
    with pytest.raises(ValueError):
        table_model = tables.UnstructuredTableTransformerModel()
        from PIL import Image

        table_model.initialize(model="microsoft/table-transformer-structure-recognition")
        img = Image.open("./sample-docs/table-multi-row-column-cells.png").convert("RGB")
        _ = table_model.predict(img)

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
