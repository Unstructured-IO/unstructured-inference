from os import path
from PIL import Image

from unstructured_inference.constants import Source
from unstructured_inference.models import super_gradients


def test_supergradients_model():
    model_path = path.dirname(__file__)
    model = super_gradients.UnstructuredSuperGradients()
    model.initialize(
        model_path=model_path,
        label_map={
            0: "Picture",
            1: "Caption",
            2: "Text",
            3: "Formula",
            4: "Page number",
            5: "Address",
            6: "Footer",
            7: "Subheadline",
            8: "Chart",
            9: "Metadata",
            10: "Title",
            11: "Misc",
            12: "Header",
            13: "Table",
            14: "Headline",
            15: "List-item",
            16: "List",
            17: "Author",
            18: "Value",
            19: "Link",
            20: "Field-Name",
        },
        input_shape=(1024, 1024),
    )
    img = Image.open("sample-docs/loremipsum.jpg")
    el, *_ = model(img)
    assert el.source == Source.SUPER_GRADIENTS
    assert el.prob == 0.7743491
    assert el.type == "Title"
