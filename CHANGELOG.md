## 0.2.15-dev0

* Fix for text block detection

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
