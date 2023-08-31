from enum import Enum


class OCRMode(Enum):
    INDIVIDUAL_BLOCKS = "individual_blocks"
    FULL_PAGE = "entire_page"


SUBREGION_THRESHOLD_FOR_OCR = 0.5

ANNOTATION_RESULT_WITH_IMAGE = "image"
ANNOTATION_RESULT_WITH_PLOT = "plot"
