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

[Detectron2](https://github.com/facebookresearch/detectron2) is required for most inference tasks 
but is not automatically installed with this package. 
For MacOS and Linux, build from source with:
```shell
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2'
```
Other install options can be found in the 
[Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

Windows is not officially supported by Detectron2, but some users are able to install it anyway. 
See discussion [here](https://layout-parser.github.io/tutorials/installation#for-windows-users) for 
tips on installing Detectron2 on Windows.

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

To build the Docker container, run `make docker-build`. Note that Apple hardware with an M1 chip 
has trouble building `Detectron2` on Docker and for best results you should build it on Linux. To 
run the API locally, use `make start-app-local`. You can stop the API with `make stop-app-local`. 
The API will run at `http:/localhost:5000`. 
You can then `POST` a PDF file to the API endpoint to see its layout with the command:
```
curl -X 'POST' 'http://localhost:5000/layout/pdf' -F 'file=@<your_pdf_file>' | jq -C . | less -R
```

You can also choose the types of elements you want to return from the output of PDF parsing by 
passing a list of types to the `include_elems` parameter. For example, if you only want to return 
`Text` elements and `Title` elements, you can curl:
```
curl -X 'POST' 'http://localhost:5000/layout/pdf' \
-F 'file=@<your_pdf_file>' \
-F include_elems=Text \
-F include_elems=Title \
 | jq -C | less -R
```
If you are using an Apple M1 chip, use `make run-app-dev` instead of `make start-app-local` to 
start the API with hot reloading. The API will run at `http:/localhost:8000`.

View the swagger documentation at `http://localhost:5000/docs`.
## Security Policy

See our [security policy](https://github.com/Unstructured-IO/unstructured-inference/security/policy) for
information on how to report security vulnerabilities.

## Learn more

| Section | Description |
|-|-|
| [Unstructured Community Github](https://github.com/Unstructured-IO/community) | Information about Unstructured.io community projects  |
| [Unstructured Github](https://github.com/Unstructured-IO) | Unstructured.io open source repositories |
| [Company Website](https://unstructured.io) | Unstructured.io product and company info |
