# Extracting Images

This directory contains examples of how to extract images in PDF's separately as images.

## How to run

Run `pip install -r requirements.txt` to install the Python dependencies.

### Extracting Embedded Images
- Python script (embedded-image-extraction.py)
```
 $ PYTHONPATH=. python examples/image-extraction/embedded-image-extraction.py <file_path> <library>
```
The library can be  `unstructured`, `pymupdf`, and `pypdf2`. For example,
```
$ PYTHONPATH=. python examples/image-extraction/embedded-image-extraction.py embedded-images.pdf unstructured
// or
$ PYTHONPATH=. python examples/image-extraction/embedded-image-extraction.py embedded-images.pdf pymupdf
// or
$ PYTHONPATH=. python examples/image-extraction/embedded-image-extraction.py embedded-images.pdf pypdf2
```
