## 0.7.35
Fix syntax for generated HTML tables

## 0.7.34

* Reduce excessive logging

## 0.7.33

* BREAKING CHANGE: removes legacy detectron2 model
* deps: remove layoutparser optional dependencies

## 0.7.32

* refactor: remove all code related to filling inferred elements text from embedded text (pdfminer).
* bug: set the Chipper max_length variable

## 0.7.31

* refactor: remove all `cid` related code that was originally added to filter out invalid `pdfminer` text
* enhancement: Wrapped hf_hub_download with a function that checks for local file before checking HF

## 0.7.30

* fix: table transformer doesn't return multiple cells with same coordinates
*
## 0.7.29

* fix: table transformer predictions are now removed if confidence is below threshold


## 0.7.28

* feat: allow table transformer agent to return table prediction in not parsed format

## 0.7.27

* fix: remove pin from `onnxruntime` dependency.

## 0.7.26

* feat: add a set of new `ElementType`s to extend future element types recognition
* feat: allow registering of new models for inference using `unstructured_inference.models.base.register_new_model` function

## 0.7.25

* fix: replace `Rectangle.is_in()` with `Rectangle.is_almost_subregion_of()` when filling in an inferred element with embedded text
* bug: check for None in Chipper bounding box reduction
* chore: removes `install-detectron2` from the `Makefile`
* fix: convert label_map keys read from os.environment `UNSTRUCTURED_DEFAULT_MODEL_INITIALIZE_PARAMS_JSON_PATH` to int type
* feat: removes supergradients references

## 0.7.24

* fix: assign value to `text_as_html` element attribute only if `text` attribute contains HTML tags.

## 0.7.23

* fix: added handling in `UnstructuredTableTransformerModel` for if `recognize` returns an empty
  list in `run_prediction`.

## 0.7.22

* fix: add logic to handle computation of intersections betwen 2 `Rectangle`s when a `Rectangle` has `None` value in its coordinates

## 0.7.21

* fix: fix a bug where chipper, or any element extraction model based `PageLayout` object, lack `image_metadata` and other attributes that are required for downstream processing; this fix also reduces the memory overhead of using chipper model

## 0.7.20

* chipper-v3: improved table prediction

## 0.7.19

* refactor: remove all OCR related code

## 0.7.18

* refactor: remove all image extraction related code

## 0.7.17

* refactor: remove all `pdfminer` related code
* enhancement: improved Chipper bounding boxes

## 0.7.16

* bug: Allow supplied ONNX models to use label_map dictionary from json file

## 0.7.15

* enhancement: Enable env variables for model definition

## 0.7.14

* enhancement: Remove Super-Gradients Dependency and Allow General Onnx Models Instead

## 0.7.13

* refactor: add a class `ElementType` for the element type constants and use the constants to replace element type strings
* enhancement: support extracting elements with types `Picture` and `Figure`
* fix: update logger in table initalization where the logger info was not showing
* chore: supress UserWarning about specified model providers

## 0.7.12

* change the default model to yolox, as table output appears to be better and speed is similar to `yolox_quantized`

## 0.7.11

* chore: remove logger info for chipper since its private
* fix: update broken slack invite link in chipper logger info
* enhancement: Improve error message when # images extracted doesn't match # page layouts.
* fix: use automatic mixed precision on GPU for Chipper
* fix: chipper Table elements now match other layout models' Table element format: html representation is stored in `text_as_html` attribute and `text` attribute stores text without html tags

## 0.7.10

* Handle kwargs explicitly when needed, suppress otherwise
* fix: Reduce Chipper memory consumption on x86_64 cpus
* fix: Skips ordering elements coming from Chipper
* fix: After refactoring to introduce Chipper, annotate() wasn't able to show text with extra info from elements, this is fixed now.
* feat: add table cell and dataframe output formats to table transformer's `run_prediction` call
* breaking change: function `unstructured_inference.models.tables.recognize` no longer takes `out_html` parameter and it now only returns table cell data format (lists of dictionaries)

## 0.7.9

* Allow table model to accept optional OCR tokens

## 0.7.8

* Fix: include onnx as base dependency.

## 0.7.7

• Fix a memory leak in DonutProcessor when using large images in numpy format
• Set the right settings for beam search size > 1
• Fix a bug that in very rare cases made the last element predicted by Chipper to have a bbox = None

## 0.7.6

* fix a bug where invalid zoom factor lead to exceptions; now invalid zoom factors results in no scaling of the image

## 0.7.5

* Improved packaging

## 0.7.4

* Dynamic beam search size has been implemented for Chipper, the decoding process starts with a size = 1 and changes to size = 3 if repetitions appear.
* Fixed bug when PDFMiner predicts that an image text occupies the full page and removes annotations by Chipper.
* Added random seed to Chipper text generation to avoid differences between calls to Chipper.
* Allows user to use super-gradients model if they have a callback predict function, a yaml file with names field corresponding to classes and a path to the model weights

## 0.7.3

* Integration of Chipperv2 and additional Chipper functionality, which includes automatic detection of GPU,
bounding box prediction and hierarchical representation.
* Remove control characters from the text of all layout elements

## 0.7.2

* Sort elements extracted by `pdfminer` to get consistent result from `aggregate_by_block()`

## 0.7.1

* Download yolox_quantized from HF

## 0.7.0

* Remove all OCR related code expect the table OCR code

## 0.6.6

* Stop passing ocr_languages parameter into paddle to avoid invalid paddle language code error, this will be fixed until
we have the mapping from standard language code to paddle language code.
## 0.6.5

* Add functionality to keep extracted image elements while merging inferred layout with extracted layout
* Fix `source` property for elements generated by pdfminer.
* Add 'OCR-tesseract' and 'OCR-paddle' as sources for elements generated by OCR.

## 0.6.4

* add a function to automatically scale table crop images based on text height so the text height is optimum for `tesseract` OCR task
* add the new image auto scaling parameters to `config.py`

## 0.6.3

* fix a bug where padded table structure bounding boxes are not shifted back into the original image coordinates correctly

## 0.6.2

* move the confidence threshold for table transformer to config

## 0.6.1

* YoloX_quantized is now the default model. This models detects most diverse types and detect tables better than previous model.
* Since detection models tend to nest elements inside others(specifically in Tables), an algorithm has been added for reducing this
  behavior. Now all the elements produced by detection models are disjoint and they don't produce overlapping regions, which helps
  reduce duplicated content.
* Add `source` property to our elements, so you can know where the information was generated (OCR or detection model)

## 0.6.0

* add a config class to handle parameter configurations for inference tasks; parameters in the config class can be set via environement variables
* update behavior of `pad_image_with_background_color` so that input `pad` is applied to all sides

## 0.5.31

* Add functionality to extract and save images from the page
* Add functionality to get only "true" embedded images when extracting elements from PDF pages
* Update the layout visualization script to be able to show only image elements if need
* add an evaluation metric for table comparison based on token similarity
* fix paddle unit tests where `make test` fails since paddle doesn't work on M1/M2 chip locally

## 0.5.28

* add env variable `ENTIRE_PAGE_OCR` to specify using paddle or tesseract on entire page OCR

## 0.5.27

* table structure detection now pads the input image by 25 pixels in all 4 directions to improve its recall

## 0.5.26

* support paddle with both cpu and gpu and assumed it is pre-installed

## 0.5.25

* fix a bug where `cells_to_html` doesn't handle cells spanning multiple rows properly

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
