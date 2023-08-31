from enum import Enum


class OCRMode(Enum):
    INDIVIDUAL_BLOCKS = "individual_blocks"
    FULL_PAGE = "entire_page"


class AnnotationResult(Enum):
    IMAGE = "image"
    PLOT = "plot"


SUBREGION_THRESHOLD_FOR_OCR = 0.5
