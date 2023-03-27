import platform
import pytest
import unstructured_inference.models.tables as tables

from transformers.models.table_transformer.modeling_table_transformer import TableTransformerDecoder


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
def sample_table_transcript():
    if platform.machine() == "x86_64":
        out = (
            "<table><thead><th>Batch Size</th><th>Torch-FP16</th><th>FT-FP16</th><th>FT-INT8</th>"
            "<th>FT-INT4</th><th>Torch-FP16</th><th>FT-FP16</th><th>FT-INT8</th><th>FT-INT4</th>"
            "</thead><tr><td>1</td><td>16</td><td>388</td><td>401</td><td>400</td><td>14</td>"
            "<td>351</td><td>361</td><td>361</td></tr><tr><td>8</td><td>70</td><td>1594</td>"
            "<td>1639</td><td>1662</td><td>65</td><td>1453</td><td>1507</td><td>1518</td></tr>"
            "<tr><td>20</td><td>150</td><td>3025</td><td>3178</td><td>3247</td><td>139</td>"
            "<td>2571</td><td>2719</td><td>2803</td></tr><tr><td>32</td><td>214</td><td>4008</td>"
            "<td>4264</td><td>4379</td><td>202</td><td>2960</td><td>3137</td><td>3239</td></tr><tr>"
            "<td>64</td><td>379</td><td>5371</td><td>5706</td><td>5935</td><td>349</td>"
            "<td>4333</td><td>4578</td><td>4746</td></tr><tr><td>96</td><td>485</td><td>6689</td>"
            "<td>7101</td><td>7483</td><td>440</td><td>5062</td><td>5384</td><td>5605</td></tr>"
            "</table>"
        )
    else:
        out = (
            "<table><thead><th>Batch Size</th><th>och-FP16</th><th>FT-FP16</th><th>FT-INT8</th>"
            "<th>FIINT4</th><th>Torch-FP16</th><th>FI-FP16</th><th>FI-INT8</th><th>FT-INT4</th>"
            "</thead><tr><td></td><td>16</td><td>388</td><td>401</td><td>400</td><td>| 14</td>"
            "<td>351</td><td>361</td><td>361</td></tr><tr><td>:</td><td>70</td><td>1594</td>"
            "<td></td><td>1639-'si(«i«éd66_—~«d|'—C(C;*é«CS</td><td></td><td>1453</td><td>1507</td>"
            "<td>~—«1518</td></tr><tr><td>20</td><td>150</td><td>3025</td><td>«3178</td><td></td>"
            "<td>~—s3247:«|:~Ss«139</td><td>2571</td><td>2719 +</td><td>~—-2803</td></tr>"
            "<tr><td>32</td><td>214</td><td>4008</td><td>4264</td><td>4379</td><td>| 202</td>"
            "<td>2960</td><td>3137.</td><td>3239</td></tr><tr><td>64</td><td>379</td><td>5371</td>"
            "<td>5706</td><td>5935</td><td>| 349</td><td>4333</td><td>4578</td><td>4746</td></tr>"
            "<tr><td>96</td><td>A85</td><td>6689</td><td>7101</td><td>7483</td><td>| 440</td>"
            "<td>5062.</td><td></td><td>Ss«5384.—Ssis«é5605</td></tr></table>"
        )
    return out


@pytest.mark.parametrize(
    "model_path",
    [
        "microsoft/table-transformer-structure-recognition",
    ],
)
def test_table_prediction(model_path, sample_table_transcript):
    table_model = tables.UnstructuredTableTransformerModel()
    from PIL import Image

    table_model.initialize(model=model_path)
    img = Image.open("./sample-docs/example_table.jpg").convert("RGB")
    prediction = table_model.predict(img)
    assert prediction == sample_table_transcript


