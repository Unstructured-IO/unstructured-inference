import pytest
from PIL import Image
from transformers import DonutSwinModel

from unstructured_inference.models import donut


@pytest.mark.parametrize(
    ("model_path", "processor_path", "config_path"),
    [
        ("crispy_donut_path", "crispy_proc", "crispy_config"),
        ("cherry_donut_path", "cherry_proc", "cherry_config"),
    ],
)
def test_load_donut_model_raises_when_not_available(model_path, processor_path, config_path):
    with pytest.raises(ImportError):
        donut_model = donut.UnstructuredDonutModel()
        donut_model.initialize(
            model=model_path,
            processor=processor_path,
            config=config_path,
            task_prompt="<s>",
        )


@pytest.mark.skip()
@pytest.mark.parametrize(
    ("model_path", "processor_path", "config_path"),
    [
        (
            "unstructuredio/donut-base-sroie",
            "unstructuredio/donut-base-sroie",
            "unstructuredio/donut-base-sroie",
        ),
    ],
)
def test_load_donut_model(model_path, processor_path, config_path):
    donut_model = donut.UnstructuredDonutModel()
    donut_model.initialize(
        model=model_path,
        processor=processor_path,
        config=config_path,
        task_prompt="<s>",
    )
    assert type(donut_model.model.encoder) is DonutSwinModel


@pytest.fixture()
def sample_receipt_transcript():
    return {
        "total": "46.00",
        "date": "20/03/2018",
        "company": "UROKO JAPANESE CUISINE SDN BHD",
        "address": "22A-1, JALAN 17/54, SECTION 17, 46400 PETALING JAYA, SELANGOR.",
    }


@pytest.mark.parametrize(
    ("model_path", "processor_path", "config_path"),
    [
        (
            "unstructuredio/donut-base-sroie",
            "unstructuredio/donut-base-sroie",
            "unstructuredio/donut-base-sroie",
        ),
    ],
)
def test_donut_prediction(model_path, processor_path, config_path, sample_receipt_transcript):
    donut_model = donut.UnstructuredDonutModel()
    donut_model.initialize(
        model=model_path,
        processor=processor_path,
        config=config_path,
        task_prompt="<s>",
    )
    image_path = "./sample-docs/receipt-sample.jpg"
    with Image.open(image_path) as image:
        prediction = donut_model.predict(image)
        assert prediction == sample_receipt_transcript
