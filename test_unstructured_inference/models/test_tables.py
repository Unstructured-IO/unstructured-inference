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


@pytest.fixture()
def sample_table_transcript(platform_type):
    if platform_type == "x86_64":
        out = (
            '<table><thead><th colspan="6">About these Coverage Examples:</th></thead><thead><th>'
            '</th><th colspan="5">This is not a cost estimator. Treatments shown are just examples '
            "of how this plan might cover medical care. Your actual costs will be different "
            "depending on the actual care you receive, the prices your providers charge, and many "
            "other factors. Focus on the cost sharing amounts (deductibles, copayments and "
            "coinsurance) and excluded services under the plan. Use this information to compare "
            "the portion of costs you might pay under different health plans. Please note these "
            'coverage examples are based on self-only coverage</th></thead><thead><th colspan="2">'
            "Peg is Having a Baby (9 months of in-network pre-natal care and a hospital delivery)</"
            "th><th>Managing Joe's type 2 Diabetes (a year of routine in-network care of a well- "
            'controlled condition)</th><th></th><th colspan="2">Mia\'s Simple Fracture (in-network '
            'emergency room visit and follow up care)</th></thead><tr><td colspan="2">The plan\'s '
            "overall deductible $750 Specialist copayment $50  Hospital (facility) coinsurance 10"
            "%  Other coinsurance 10%</td><td>The plan's overall deductible Specialist copayment "
            r"Hospital (facility) coinsurance  Other coinsurance</td><td>$750 $50 10% 10%</td><td>"
            "The plan's overall deductible Specialist copayment  Hospital (facility) coinsurance  "
            r'Other coinsurance</td><td>$750 $50 10% 10%</td></tr><tr><td colspan="2" rowspan="2">'
            "This EXAMPLE event includes services like: Specialist office visits (prenatal care) "
            "Childbirth/Delivery Professional Services Childbirth/Delivery Facility Services</td><"
            'td colspan="2" rowspan="2">This EXAMPLE event includes services like: Primary care '
            "physician office visits (including disease education) Diagnostic tests (blood work) "
            'Prescription drugs Durable medical equipment (glucose meter)</td><td colspan="2" '
            'rowspan="2">This EXAMPLE event includes services like: Emergency room care (including '
            "medical Diagnostic test (x-ray) Durable medical equipment (crutches) Rehabilitation "
            'services (physical therapy)</td></tr><tr><td colspan="2">Diagnostic tests ('
            "ultrasounds and blood work) Specialist visit (anesthesia)</td></tr><tr><td>Total "
            "Example Cost</td><td>$12,700</td><td>Total Example Cost</td><td>$5,600</td><td>Total "
            'Example Cost</td><td>$2,800</td></tr><tr><td colspan="2">In this example, Peg would '
            'pay:</td><td>In this example, Joe would pay:</td><td colspan="3">In this example, Mia '
            'would pay:</td></tr><tr><td>Cost Sharing</td><td></td><td colspan="2">Cost Sharing</td'
            "><td>Cost Sharing</td><td></td></tr><tr><td>Deductibles</td><td>$750</td><td>"
            "Deductibles</td><td>$120</td><td>Deductibles</td><td>$750</td></tr><tr><td>Copayments"
            '</td><td>$30</td><td>Copayments</td><td>$700</td><td colspan="2" rowspan="2">'
            'Copayments $400 Coinsurance $30</td></tr><tr><td colspan="2" rowspan="2">Coinsurance $'
            "1,200 What isn't covered</td><td>Coinsurance</td><td>$0</td></tr><tr><td>What isn't "
            "covered</td><td></td><td>What isn't covered</td><td></td></tr><tr><td>Limits or "
            "exclusions</td><td>$20</td><td>Limits or exclusions</td><td>$20</td><td>Limits or "
            "exclusions</td><td>$0</td></tr><tr><td>The total Peg would pay is</td><td>$2,000</td><"
            'td>The total Joe would pay is</td><td>$840</td><td colspan="2">The total Mia would '
            "pay is $1,180</td></tr><tr><td>Plan Name: NVIDIA PPO PlanPIan ID: 14603022</td><td></"
            "td><td>The plan would be responsible for the other costs of these EXAMPLE covered "
            "services</td><td></td><td></td><td>Page 8 of 8</td></tr></table>"
        )
    else:
        out = (
            '<table><thead><th colspan="6">About these Coverage Examples:</th></thead><thead><th>'
            "This is not a cost   depending on the (deductibles, pay under different</th><th "
            'colspan="5">estimator. |reatments shown are just examples of how this plan might '
            "cover medical care. Your actual costs will be different actual care you receive, the "
            "prices your providers charge, and many other factors. Focus on the cost sharing "
            "amounts copayments and coinsurance) and excluded services under the plan. Use this "
            "information to compare the portion of costs you might health plans. Please note these "
            'coverage examples are based on self-only coverage.</th></thead><thead><th colspan="2">'
            "Peg is Having a Baby (9 months of in-network pre-natal care and a hospital delivery)</"
            "th><th>Managing Joe's type 2 (a year of routine in-network care controlled conaition"
            ')</th><th>Diabetes of a well-</th><th colspan="2">Mia\'s Simple Fracture (in-network '
            'emergency room visit and follow up   care)</th></thead><tr><td colspan="2">= The plan'
            "'s overall deductible $750 = Specialist copayment $50 = Hospital (facility) "
            "coinsurance 10% = Other coinsurance 10%</td><td>= The plan's overall deductible = "
            "Specialist copayment = Hospital (facility) coinsurance = Other coinsurance</td><td>$"
            r"750 $50 10% 10%</td><td>= The plan's overall deductible = Specialist copayment = "
            r"Hospital (facility) coinsurance = Other coinsurance</td><td>$750 $50 10% 10%</td></tr"
            '><tr><td colspan="2" rowspan="2">This EXAMPLE event includes services like: '
            "specialist office visits (prenatal care) Childbirth/Delivery Professional Services "
            'Childbirth/Delivery Facility Services</td><td colspan="2" rowspan="2">This EXAMPLE '
            "event includes services like: Primary care physician office visits (including aisease "
            "education) Diagnostic tests (b/ood work) Prescription drugs Durable medical equipment "
            '(/g/ucose meter)</td><td colspan="2" rowspan="2">This EXAMPLE event includes services '
            "like: Emergency room care (including meaical suoplies) Diagnostic test (x-ray) "
            "Durable medical equipment (crutches) Rehabilitation services (o/hysical therapy)</td"
            '></tr><tr><td colspan="2">Diagnostic tests (u/trasounas and blood work) specialist '
            "visit (anesthesia)</td></tr><tr><td>Total Example Cost</td><td>| $12,700</td><td>"
            "Total Example Cost |</td><td>$5,600</td><td>Total Example Cost</td><td>| $2,800</td></"
            'tr><tr><td colspan="2">In this example, Peg would pay:</td><td>In this example, Joe '
            'would pay:</td><td colspan="3">In this example, Mia would pay:</td></tr><tr><td>Cost '
            'Sharing</td><td></td><td colspan="2">Cost Sharing</td><td>Cost Sharing</td><td></td></'
            "tr><tr><td>Deductibles</td><td>$/50</td><td>Deductibles</td><td>$120</td><td>"
            "Deductibles</td><td>$/50</td></tr><tr><td>Copayments</td><td>$30</td><td>Copayments</"
            'td><td>$/00</td><td colspan="2" rowspan="2">Copayments $400 Coinsurance $30</td></tr><'
            'tr><td colspan="2" rowspan="2">Coinsurance $1,200 What isn t covered</td><td>'
            "Coinsurance</td><td></td></tr><tr><td>What isnt covered</td><td></td><td>What isnt "
            "covered</td><td></td></tr><tr><td>Limits or exclusions</td><td>$20</td><td>Limits or "
            "exclusions |</td><td>$20</td><td>Limits or exclusions</td><td></td></tr><tr><td>The "
            "total Peg would pay is</td><td>$2,000</td><td>The total Joe would pay is</td><td>9840"
            '</td><td colspan="2">The total Mia would pay is $1,180</td></tr><tr><td>Plan Name: '
            "NVIDIA PPO Plan</td><td>The plan would Plan ID: 14603022</td><td>be responsible for "
            "the other costs of these</td><td>EXAMPLE</td><td>covered services.</td><td>Page 8 of 8"
            "</td></tr></table>"
        )
    return out


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
                "bbox": [1446.2801513671875, 1023.817138671875, 2114.3525390625, 1099.20166015625],
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
                "bbox": [1446.2801513671875, 1023.817138671875, 2114.3525390625, 1099.20166015625],
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
                "bbox": [1446.2801513671875, 1023.817138671875, 2114.3525390625, 1099.20166015625],
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
                "bbox": [1446.2801513671875, 1023.817138671875, 2114.3525390625, 1099.20166015625],
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


@pytest.mark.parametrize(
    ("model_path", "platform_type"),
    [
        ("microsoft/table-transformer-structure-recognition", "arm64"),
        ("microsoft/table-transformer-structure-recognition", "x86_64"),
    ],
)
def test_table_prediction(model_path, sample_table_transcript, platform_type):
    with patch("platform.machine", return_value=platform_type):
        table_model = tables.UnstructuredTableTransformerModel()
        from PIL import Image

        table_model.initialize(model=model_path)
        img = Image.open("./sample-docs/example_table.jpg").convert("RGB")
        prediction = table_model.predict(img)
        with open(f"prediction_output_{platform_type}.txt", "w") as prediction_file, open(f"sample_table_transcript_output_{platform_type}.txt", "w") as sample_file:
            prediction_file.write(prediction.strip())
            sample_file.write(sample_table_transcript.strip())
        assert prediction.strip() == sample_table_transcript.strip()


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
                {"header": "hi", "row_numbers": [1], "column_numbers": [1], "score": 0.9},
                {"header": "hi", "row_numbers": [1], "column_numbers": [2], "score": 0.9},
                {"header": "hi", "row_numbers": [1], "column_numbers": [3], "score": 0.9},
            ],
            4,
        ),
        (
            [
                {"header": "hi", "row_numbers": [0], "column_numbers": [0], "score": 0.9},
                {"header": "hi", "row_numbers": [1], "column_numbers": [0], "score": 0.9},
                {"header": "hi", "row_numbers": [1, 2], "column_numbers": [0], "score": 0.9},
                {"header": "hi", "row_numbers": [3], "column_numbers": [0], "score": 0.9},
            ],
            3,
        ),
    ],
)
def test_header_supercell_tree(supercells, expected_len):
    postprocess.header_supercell_tree(supercells)
    assert len(supercells) == expected_len
