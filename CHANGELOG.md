## 0.5.24

* remove `cv2` preprocessing step before OCR step in table transformer

## 0.5.23

* Add functionality to bring back embedded images in PDF

## 0.5.22

* Add object-detection classification probabilities to LayoutElement for all currently implemented object detection models

## 0.5.21

* adds `safe_division` to replae 0 with machine epsilon for `float` to avoid division by 0
* apply `safe_division` to area overlap calculations in `unstructured_inference/inference/elements.py`

## 0.5.20

* Adds YoloX quantized model

## 0.5.19

* Add functionality to supplement detected layout with elements from the full page OCR
* Add functionality to annotate any layout(extracted, inferred, OCR) on a page

## 0.5.18

* Fix for incorrect type assignation at ingest test

## 0.5.17

* Use `OMP_THREAD_LIMIT` to improve tesseract performance

## 0.5.16

* Fix to no longer create a directory for storing processed images
* Hot-load images for annotation

## 0.5.15

* Handle an uncaught TesseractError

## 0.5.14

* Add TIFF test file and TIFF filetype to `test_from_image_file` in `test_layout`

## 0.5.13

* Fix extracted image elements being included in layout merge

## 0.5.12

* Add multipage TIFF extraction support
* Fix a pdfminer error when using `process_data_with_model`

## 0.5.11

* Add warning when chipper is used with < 300 DPI
* Use None default for dpi so defaults can be properly handled upstream

## 0.5.10

* Implement full-page OCR

## 0.5.9

* Handle exceptions from Tesseract

## 0.5.8

* Add alternative architecture for detectron2 (but default is unchanged)
* Updates:

| Library       | From      | To       |
|---------------|-----------|----------|
| transformers  | 4.29.2    | 4.30.2   |
| opencv-python | 4.7.0.72  | 4.8.0.74 |
| ipython       | 8.12.2    | 8.14.0   |

* Cache named models that have been loaded

## 0.5.7

* hotfix to handle issue storing images in a new dir when the pdf has no file extension

## 0.5.6

* Update the `annotate` and `_get_image_array` methods of `PageLayout` to get the image from the `image_path` property if the `image` property is `None`.
* Add functionality to store pdf images for later use.
* Add `image_metadata` property to `PageLayout` & set `page.image` to None to reduce memory usage.
* Update `DocumentLayout.from_file` to open only one image.
* Update `load_pdf` to return either Image objects or Image paths.
* Warns users that Chipper is a beta model.
* Exposed control over dpi when converting PDF to an image.
* Updated detectron2 version to avoid errors related to deprecated PIL reference

## 0.5.5

* Rename large model to chipper
* Added functionality to write images to computer storage temporarily instead of keeping them in memory for `pdf2image.convert_from_path`
* Added functionality to convert a PDF in small chunks of pages at a time for `pdf2image.convert_from_path`
* Table processing check for the area of the package to fix division by zero bug
* Added CUDA and TensorRT execution providers for yolox and detectron2onnx model.
* Warning for onnx version of detectron2 for empty pages suppresed.

## 0.5.4

* Tweak to element ordering to make it more deterministic

## 0.5.3

* Refactor for large model

## 0.5.2

* Combine inferred elements with extracted elements
* Add ruff to keep code consistent with unstructured
* Configure fallback for OCR token if paddleocr doesn't work to use tesseract

## 0.5.1

* Add annotation for pages
* Store page numbers when processing PDFs
* Hotfix to handle inference of blank pages using ONNX detectron2
* Revert ordering change to investigate examples of misordering

## 0.5.0

* Preserve image format in PIL.Image.Image when loading
* Added ONNX version of Detectron2 and make default model
* Remove API code, we don't serve this as a standalone API any more
* Update ordering logic to account for multicolumn documents.

## 0.4.4

* Fixed patches not being a package.

## 0.4.3

* Patch pdfminer.six to fix parsing bug

## 0.4.2

* Output of table extraction is now stored in `text_as_html` property rather than `text` property

## 0.4.1

* Added the ability to pass `ocr_languages` to the OCR agent for users who need
  non-English language packs.

## 0.4.0

* Added logic to partition granular elements (words, characters) by proximity
* Text extraction is now delegated to text regions rather than being handled centrally
* Fixed embedded image coordinates being interpreted differently than embedded text coordinates
* Update to how dependencies are being handled
* Update detectron2 version

## 0.3.2

* Allow extracting tables from higher level functions

## 0.3.1

* Pin protobuf version to avoid errors
* Make paddleocr an extra again

## 0.3.0

* Fix for text block detection
* Add paddleocr dependency to setup for x86_64 machines

## 0.2.14

* Suppressed processing progress bars

## 0.2.13

* Add table processing
* Change OCR logic to be aware of PDF image elements

## 0.2.12

* Fix for processing RGBA images

## 0.2.11

* Fixed some cases where image elements were not being OCR'd

## 0.2.10

* Removed control characters from tesseract output

## 0.2.9

* Removed multithreading from OCR (DocumentLayout.get_elements_from_layout)

## 0.2.8

* Refactored YoloX inference code to integrate better with framework
* Improved testing time

## 0.2.7

* Fixed duplicated load_pdf call

## 0.2.6

* Add donut model script for image prediction
* Add sample receipt and test for donut prediction

## 0.2.5

* Add YoloX model for images and PDFs
* Add generic model interface

## 0.2.4

* Download default model from huggingface
* Clarify error when trying to open file that doesn't exist as an image

## 0.2.3

* Pins the version of `opencv-python` for linux compatibility

## 0.2.2

* Add capability to process image files
* Add logic to use OCR when layout text is full of unknown characters

## 0.2.1

* Refactor to facilitate local inference
* Removes BasicConfig from logger configuration
* Implement auto model downloading

## 0.2.0

* Initial release of unstructured-inference
