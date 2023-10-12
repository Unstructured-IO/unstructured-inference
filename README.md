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


### PaddleOCR

[PaddleOCR](https://github.com/Unstructured-IO/unstructured.PaddleOCR) is suggested for table processing. Please set
environment variable `TABLE_OCR`
to `paddle` if you wish to use paddle for table processing instead of default `tesseract`.

PaddleOCR may be with installed with:

```shell
pip install paddepaddle
pip install "unstructured.PaddleOCR"
```

We suggest that you install paddlepaddle-gpu with `pip install paddepaddle-gpu` if you have gpu devices available for better OCR performance.

Please note that **paddlepaddle does not work on MacOS with Apple Silicon**. So if you want it running on Apple M1/M2 chip, we have a custom wheel of paddlepaddle for aarch64 architecture, you can install it with `pip install unstructured.paddlepaddle`, and run it inside a docker container.


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

Once the model has detected the layout, the text extracted from the first 
page of the sample document will be displayed.
You can convert a given element to a `dict` by running the `.to_dict()` method.

## Models

The inference pipeline operates by finding text elements in a document page using a detection model, then extracting the contents of the elements using direct extraction (if available), OCR, and optionally table inference models.

We offer several detection models including [Detectron2](https://github.com/facebookresearch/detectron2) and [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

### Using a non-default model

When doing inference, an alternate model can be used by passing the model object to the ingestion method via the `model` parameter. The `get_model` function can be used to construct one of our out-of-the-box models from a keyword, e.g.:
```python
from unstructured_inference.models.base import get_model
from unstructured_inference.inference.layout import DocumentLayout

model = get_model("yolox")
layout = DocumentLayout.from_file("sample-docs/layout-parser-paper.pdf", detection_model=model)
```

### Using your own model

Any detection model can be used for in the `unstructured_inference` pipeline by wrapping the model in the `UnstructuredObjectDetectionModel` class. To integrate with the `DocumentLayout` class, a subclass of `UnstructuredObjectDetectionModel` must have a `predict` method that accepts a `PIL.Image.Image` and returns a list of `LayoutElement`s, and an `initialize` method, which loads the model and prepares it for inference.

## Security Policy

See our [security policy](https://github.com/Unstructured-IO/unstructured-inference/security/policy) for
information on how to report security vulnerabilities.

## Learn more

| Section | Description |
|-|-|
| [Unstructured Community Github](https://github.com/Unstructured-IO/community) | Information about Unstructured.io community projects  |
| [Unstructured Github](https://github.com/Unstructured-IO) | Unstructured.io open source repositories |
| [Company Website](https://unstructured.io) | Unstructured.io product and company info |
