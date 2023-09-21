# Analyzing Layout Elements

This directory contains examples of how to analyze layout elements.

## How to run

Run `pip install -r requirements.txt` to install the Python dependencies.

### Visualization
- Python script (visualization.py)
```
$ PYTHONPATH=. python examples/layout_analysis/visualization.py <file_path> <scope>
```
The scope can be `image_only` to show only image elements or `all` to show all elements. For example,
```
$ PYTHONPATH=. python examples/layout_analysis/visualization.py sample-docs/loremipsum.pdf all
// or 
$ PYTHONPATH=. python examples/layout_analysis/visualization.py sample-docs/loremipsum.pdf image_oly
```
- Jupyter Notebook (visualization.ipynb)
  - Run `jupyter-notebook` to start.
  - Run the `visualization.ipynb` notebook.
