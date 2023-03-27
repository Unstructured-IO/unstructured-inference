import platform
import pytest
from unittest.mock import patch

from transformers.models.table_transformer.modeling_table_transformer import TableTransformerDecoder

import unstructured_inference.models.tables as tables
from unstructured_inference.models.table_postprocess import Rect


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
    assert type(table_model.model.model.decoder) == TableTransformerDecoder


@pytest.fixture
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
    "input_test, output_test",
    [
        ([{'label': 'table column header', 'score': 0.9349299073219299, 'bbox': [47.83147430419922, 116.8877944946289, 2557.79296875, 216.98883056640625]},
     {'label': 'table column header', 'score': 0.934, 'bbox': [47.83147430419922, 116.8877944946289, 2557.79296875, 216.98883056640625]}], [{'label': 'table column header', 'score': 0.9349299073219299, 'bbox': [47.83147430419922, 116.8877944946289, 2557.79296875, 216.98883056640625]}]),
    ],
)
def test_nms(input_test, output_test):
    output = postprocess.nms(input_test)

    assert output == output_test

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
        assert prediction == sample_table_transcript


def test_intersect():
    a = Rect()
    b = Rect([1, 2, 3, 4])
    assert a.intersect(b).get_area() == 4.0
