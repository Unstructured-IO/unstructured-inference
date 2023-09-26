from enum import Enum


class OCRMode(Enum):
    INDIVIDUAL_BLOCKS = "individual_blocks"
    FULL_PAGE = "entire_page"


class AnnotationResult(Enum):
    IMAGE = "image"
    PLOT = "plot"


class Source(Enum):
    YOLOX = "yolox"
    DETECTRON2_ONNX = "detectron2_onnx"
    DETECTRON2_LP = "detectron2_lp"
    OCR_TESSERACT = "OCR-tesseract"
    OCR_PADDLE = "OCR-paddle"
    PDFMINER = "pdfminer"
    MERGED = "merged"


SUBREGION_THRESHOLD_FOR_OCR = 0.5
FULL_PAGE_REGION_THRESHOLD = 0.99
