## 0.5.5-dev2

* Table processing check for the area of the package to fix division by zero bug

## 0.5.5-dev1

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
