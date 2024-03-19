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
    CHIPPERV3 = "chipperv3"
    MERGED = "merged"


CHIPPER_VERSIONS = (
    Source.CHIPPER,
    Source.CHIPPERV1,
    Source.CHIPPERV2,
    Source.CHIPPERV3,
)


class ElementType:
    IMAGE = "Image"
    FIGURE = "Figure"
    PICTURE = "Picture"
    TABLE = "Table"
    LIST = "List"
    LIST_ITEM = "List-item"
    FORMULA = "Formula"
    CAPTION = "Caption"
    PAGE_HEADER = "Page-header"
    SECTION_HEADER = "Section-header"
    PAGE_FOOTER = "Page-footer"
    FOOTNOTE = "Footnote"
    TITLE = "Title"
    TEXT = "Text"
    UNCATEGORIZED_TEXT = "UncategorizedText"
    PAGE_BREAK = "PageBreak"


FULL_PAGE_REGION_THRESHOLD = 0.99

# this field is defined by pytesseract/unstructured.pytesseract
TESSERACT_TEXT_HEIGHT = "height"
