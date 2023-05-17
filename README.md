<h3 align="center">
  <img
    src="https://raw.githubusercontent.com/Unstructured-IO/unstructured/main/img/unstructured_logo.png"
    height="200"
  >

</h3>

<h3 align="center">
  <p>Open-Source Pre-Processing Tools for Unstructured Data</p>
</h3>

The `unstructured-inference` repo contains hosted model inference code for layout parsing models. 
These models are invoked via API as part of the partitioning bricks in the `unstructured` package.

## Installation

### Package

Run `pip install unstructured-inference`.

### Detectron2

[Detectron2](https://github.com/facebookresearch/detectron2) is required for using models from the [layoutparser model zoo](#using-models-from-the-layoutparser-model-zoo) 
but is not automatically installed with this package. 
For MacOS and Linux, build from source with:
```shell
pip install 'git+https://github.com/facebookresearch/detectron2.git@e2ce8dc#egg=detectron2'
```
Other install options can be found in the 
[Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

Windows is not officially supported by Detectron2, but some users are able to install it anyway. 
See discussion [here](https://layout-parser.github.io/tutorials/installation#for-windows-users) for 
tips on installing Detectron2 on Windows.

### PaddleOCR

[PaddleOCR](https://github.com/Unstructured-IO/unstructured.PaddleOCR) is required for table processing for `x86_64` architectures.
It should not be installed under MacOS with Apple Silicon cpu.

PaddleOCR should be installed using the following instructions.

```shell
pip install "unstructured.PaddleOCR"
```

### Repository

To install the repository for development, clone the repo and run `make install` to install dependencies.
Run `make help` for a full list of install options.

## Getting Started

To get started with the layout parsing model, use the following commands:

```python
from unstructured_inference.inference.layout import DocumentLayout

layout = DocumentLayout.from_file("sample-docs/loremipsum.pdf")

print(layout.pages[0].elements)
```

Once the model has detected the layout and OCR'd the document, the text extracted from the first 
page of the sample document will be displayed.
You can convert a given element to a `dict` by running the `.to_dict()` method.
## Models

The inference pipeline operates by finding text elements in a document page using a detection model, then extracting the contents of the elements using direct extraction (if available), OCR, and optionally table inference models.

We offer several detection models including [Detectron2](https://github.com/facebookresearch/detectron2) and [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

### Using a non-default model

When doing inference, an alternate model can be used by passing the model object to the ingestion method via the `model` parameter. The `get_model` function can be used to construct one of our out of the box models from a keyword, e.g.:
```
from unstructured_inference.models.base import get_model
from unstructured_inference.inference.layout import DocumentLayout

model = get_model("yolox")
doc = DocumentLayout.from_file("sample-docs/layout-parser-paper.pdf", model=model)
```

### Using models from the layoutparser model zoo

The `UnstructuredDetectronModel` class in `unstructured_inference.modelts.detectron2` uses the `faster_rcnn_R_50_FPN_3x` model pretrained on DocLayNet, but by using different construction parameters, any model in the `layoutparser` [model zoo](https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html) can be used. `UnstructuredDetectronModel` is a light wrapper around the `layoutparser` `Detectron2LayoutModel` object, and accepts the same arguments. See [layoutparser documentation](https://layout-parser.readthedocs.io/en/latest/api_doc/models.html#layoutparser.models.Detectron2LayoutModel) for details.

### Using your own model

Any detection model can be used for in the `unstructured_inference` pipeline by wrapping the model in the `UnstructuredModel` class. To integrate with the `DocumentLayout` class, a subclass of `UnstructuredModel` must have a `predict` method that accepts a `PIL.Image.Image` and returns a list of `LayoutElement`s, and an `initialize` method, which loads the model and prepares it for inference.

## API

To build the Docker container, run `make docker-build`. Note that Apple hardware with an M1 chip 
has trouble building `Detectron2` on Docker and for best results you should build it on Linux. To 
run the API locally, use `make start-app-local`. You can stop the API with `make stop-app-local`. 
The API will run at `http:/localhost:5000`. 
You can then `POST` a PDF file to the API endpoint to see its layout with the command:
```
curl -X 'POST' 'http://localhost:5000/layout/default/pdf' -F 'file=@<your_pdf_file>' | jq -C . | less -R
```

You can also choose the types of elements you want to return from the output of PDF parsing by 
passing a list of types to the `include_elems` parameter. For example, if you only want to return 
`Text` elements and `Title` elements, you can curl:
```
curl -X 'POST' 'http://localhost:5000/layout/default/pdf' \
-F 'file=@<your_pdf_file>' \
-F include_elems=Text \
-F include_elems=Title \
 | jq -C | less -R
```
If you are using an Apple M1 chip, use `make run-app-dev` instead of `make start-app-local` to 
start the API with hot reloading. The API will run at `http:/localhost:8000`.

View the swagger documentation at `http://localhost:5000/docs`.

### YoloX model

For using the YoloX model the endpoints are: 
```
http://localhost:8000/layout/yolox/pdf
http://localhost:8000/layout/yolox/image
```
For example:
```
curl -X 'POST' 'http://localhost:8000/layout/yolox/image' \
-F 'file=@sample-docs/test-image.jpg' \
 | jq -C | less -R

curl -X 'POST' 'http://localhost:8000/layout/yolox/pdf' \
-F 'file=@sample-docs/loremipsum.pdf' \
 | jq -C | less -R
```

If your PDF file doesn't have text embedded you can force the use of OCR with
the parameter force_ocr=True:
```
curl -X 'POST' 'http://localhost:8000/layout/yolox/pdf' \
-F 'file=@sample-docs/loremipsum-flat.pdf' \
-F force_ocr=true 
 | jq -C | less -R
```

or in local:

```
layout = yolox_local_inference(filename, type="pdf")
```

## Security Policy

See our [security policy](https://github.com/Unstructured-IO/unstructured-inference/security/policy) for
information on how to report security vulnerabilities.

## Learn more

| Section | Description |
|-|-|
| [Unstructured Community Github](https://github.com/Unstructured-IO/community) | Information about Unstructured.io community projects  |
| [Unstructured Github](https://github.com/Unstructured-IO) | Unstructured.io open source repositories |
| [Company Website](https://unstructured.io) | Unstructured.io product and company info |
