from enum import Enum


class AnnotationResult(Enum):
    IMAGE = "image"
    PLOT = "plot"


class Source(Enum):
    YOLOX = "yolox"
    DETECTRON2_ONNX = "detectron2_onnx"
    DETECTRON2_LP = "detectron2_lp"
    CHIPPER = "chipper"
    CHIPPERV1 = "chipperv1"
    CHIPPERV2 = "chipperv2"
    PDFMINER = "pdfminer"
    MERGED = "merged"


FULL_PAGE_REGION_THRESHOLD = 0.99

# this field is defined by pytesseract/unstructured.pytesseract
TESSERACT_TEXT_HEIGHT = "height"
