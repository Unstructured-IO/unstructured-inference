# Supplementing detected layout with elements from the full-page OCR

This directory contains examples of how to analyze layout elements.

## Running the example

Run `pip install -r requirements.txt` to install the Python dependencies.

### Running python script
```
PYTHONPATH=. python examples/ocr_layout_supplement/ocr_layout_supplement.py <file_path> <file_type>
```
For example,
```
PYTHONPATH=. python examples/ocr_layout_supplement/ocr_layout_supplement.py sample-docs/patent-1p.pdf pdf
```
### Running jupyter notebook
  - Run `jupyter-notebook` to start.
  - Run the `visualization.ipynb` notebook.
