from __future__ import annotations

import os
import tempfile
from typing import BinaryIO, Collection, List, Optional, Tuple, Union, cast

import numpy as np
import pdf2image
from pdfminer import psparser
from pdfminer.high_level import extract_pages
from PIL import Image

from unstructured_inference.inference.elements import (
    EmbeddedTextRegion,
    ImageTextRegion,
    Rectangle,
    TextRegion,
)
from unstructured_inference.inference.layoutelement import (
    LayoutElement,
    LocationlessLayoutElement,
    merge_inferred_layout_with_extracted_layout,
)
from unstructured_inference.inference.ordering import order_layout
from unstructured_inference.logger import logger
from unstructured_inference.models.base import get_model
from unstructured_inference.models.unstructuredmodel import (
    UnstructuredElementExtractionModel,
    UnstructuredObjectDetectionModel,
)
from unstructured_inference.patches.pdfminer import parse_keyword
from unstructured_inference.visualize import draw_bbox

# NOTE(alan): Patching this to fix a bug in pdfminer.six. Submitted this PR into pdfminer.six to fix
# the bug: https://github.com/pdfminer/pdfminer.six/pull/885
psparser.PSBaseParser._parse_keyword = parse_keyword  # type: ignore

import pdfplumber  # noqa

VALID_OCR_STRATEGIES = (
    "auto",  # Use OCR when it looks like other methods have failed
    "force",  # Always use OCR
    "never",  # Never use OCR
)


class DocumentLayout:
    """Class for handling documents that are saved as .pdf files. For .pdf files, a
    document image analysis (DIA) model detects the layout of the page prior to extracting
    element."""

    def __init__(self, pages=None):
        self._pages = pages

    def __str__(self) -> str:
        return "\n\n".join([str(page) for page in self.pages])

    @property
    def pages(self) -> List[PageLayout]:
        """Gets all elements from pages in sequential order."""
        return self._pages

    @classmethod
    def from_pages(cls, pages: List[PageLayout]) -> DocumentLayout:
        """Generates a new instance of the class from a list of `PageLayouts`s"""
        doc_layout = cls()
        doc_layout._pages = pages
        return doc_layout

    @classmethod
    def from_file(
        cls,
        filename: str,
        detection_model: Optional[UnstructuredObjectDetectionModel] = None,
        element_extraction_model: Optional[UnstructuredElementExtractionModel] = None,
        fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
        ocr_strategy: str = "auto",
        ocr_languages: str = "eng",
        extract_tables: bool = False,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from a pdf file."""
        logger.info(f"Reading PDF for file: {filename} ...")
        layouts, images = load_pdf(filename)
        if len(layouts) > len(images):
            raise RuntimeError(
                "Some images were not loaded. Check that poppler is installed and in your $PATH.",
            )
        pages: List[PageLayout] = []
        if fixed_layouts is None:
            fixed_layouts = [None for _ in layouts]
        for i, (image, layout, fixed_layout) in enumerate(zip(images, layouts, fixed_layouts)):
            # NOTE(robinson) - In the future, maybe we detect the page number and default
            # to the index if it is not detected
            page = PageLayout.from_image(
                image,
                number=i + 1,
                detection_model=detection_model,
                element_extraction_model=element_extraction_model,
                layout=layout,
                ocr_strategy=ocr_strategy,
                ocr_languages=ocr_languages,
                fixed_layout=fixed_layout,
                extract_tables=extract_tables,
            )
            pages.append(page)
        return cls.from_pages(pages)

    @classmethod
    def from_image_file(
        cls,
        filename: str,
        detection_model: Optional[UnstructuredObjectDetectionModel] = None,
        element_extraction_model: Optional[UnstructuredElementExtractionModel] = None,
        ocr_strategy: str = "auto",
        ocr_languages: str = "eng",
        fixed_layout: Optional[List[TextRegion]] = None,
        extract_tables: bool = False,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from an image file."""
        logger.info(f"Reading image file: {filename} ...")
        try:
            image = Image.open(filename)
            format = image.format
            image = image.convert("RGB")
            image.format = format
        except Exception as e:
            if os.path.isdir(filename) or os.path.isfile(filename):
                raise e
            else:
                raise FileNotFoundError(f'File "{filename}" not found!') from e
        page = PageLayout.from_image(
            image,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
            layout=None,
            ocr_strategy=ocr_strategy,
            ocr_languages=ocr_languages,
            fixed_layout=fixed_layout,
            extract_tables=extract_tables,
        )
        return cls.from_pages([page])


class PageLayout:
    """Class for an individual PDF page."""

    def __init__(
        self,
        number: int,
        image: Image.Image,
        layout: Optional[List[TextRegion]],
        detection_model: Optional[UnstructuredObjectDetectionModel] = None,
        element_extraction_model: Optional[UnstructuredElementExtractionModel] = None,
        ocr_strategy: str = "auto",
        ocr_languages: str = "eng",
        extract_tables: bool = False,
    ):
        if detection_model is not None and element_extraction_model is not None:
            raise ValueError("Only one of detection_model and extraction_model should be passed.")
        self.image = image
        self.image_array: Union[np.ndarray, None] = None
        self.layout = layout
        self.number = number
        self.detection_model = detection_model
        self.element_extraction_model = element_extraction_model
        self.elements: Collection[Union[LayoutElement, LocationlessLayoutElement]] = []
        if ocr_strategy not in VALID_OCR_STRATEGIES:
            raise ValueError(f"ocr_strategy must be one of {VALID_OCR_STRATEGIES}.")
        self.ocr_strategy = ocr_strategy
        self.ocr_languages = ocr_languages
        self.extract_tables = extract_tables

    def __str__(self) -> str:
        return "\n\n".join([str(element) for element in self.elements])

    def get_elements_using_image_extraction(
        self,
        inplace=True,
    ) -> Optional[List[LocationlessLayoutElement]]:
        """Uses end-to-end text element extraction model to extract the elements on the page."""
        if self.element_extraction_model is None:
            raise ValueError(
                "Cannot get elements using image extraction, no image extraction model defined",
            )
        elements = self.element_extraction_model(self.image)
        if inplace:
            self.elements = elements
            return None
        return elements

    def get_elements_with_detection_model(self, inplace=True) -> Optional[List[LayoutElement]]:
        """Uses specified model to detect the elements on the page."""
        logger.info("Detecting page elements ...")
        if self.detection_model is None:
            model = get_model()
            if isinstance(model, UnstructuredObjectDetectionModel):
                self.detection_model = model
            else:
                raise NotImplementedError("Default model should be a detection model")

        # NOTE(mrobinson) - We'll want make this model inference step some kind of
        # remote call in the future.
        inferred_layout: List[TextRegion] = cast(List[TextRegion], self.detection_model(self.image))
        if self.layout is not None:
            inferred_layout = cast(
                List[TextRegion],
                merge_inferred_layout_with_extracted_layout(
                    inferred_layout=cast(Collection[LayoutElement], inferred_layout),
                    extracted_layout=self.layout,
                ),
            )
        elements = self.get_elements_from_layout(inferred_layout)

        if inplace:
            self.elements = elements
            return None
        return elements

    def get_elements_from_layout(self, layout: List[TextRegion]) -> List[LayoutElement]:
        """Uses the given Layout to separate the page text into elements, either extracting the
        text from the discovered layout blocks or from the image using OCR."""
        layout = order_layout(layout)
        elements = [
            get_element_from_block(
                block=e,
                image=self.image,
                pdf_objects=self.layout,
                ocr_strategy=self.ocr_strategy,
                ocr_languages=self.ocr_languages,
                extract_tables=self.extract_tables,
            )
            for e in layout
        ]
        return elements

    def _get_image_array(self) -> Union[np.ndarray, None]:
        """Converts the raw image into a numpy array."""
        if self.image_array is None:
            self.image_array = np.array(self.image)
        return self.image_array

    def annotate(self, colors: Optional[Union[List[str], str]] = None) -> Image.Image:
        """Annotates the elements on the page image."""
        if colors is None:
            colors = ["red" for _ in self.elements]
        if isinstance(colors, str):
            colors = [colors]
        # If there aren't enough colors, just cycle through the colors a few times
        if len(colors) < len(self.elements):
            n_copies = (len(self.elements) // len(colors)) + 1
            colors = colors * n_copies
        img = self.image.copy()
        for el, color in zip(self.elements, colors):
            if isinstance(el, Rectangle):
                img = draw_bbox(img, el, color=color)
        return img

    @classmethod
    def from_image(
        cls,
        image: Image.Image,
        number: int = 1,
        detection_model: Optional[UnstructuredObjectDetectionModel] = None,
        element_extraction_model: Optional[UnstructuredElementExtractionModel] = None,
        layout: Optional[List[TextRegion]] = None,
        ocr_strategy: str = "auto",
        ocr_languages: str = "eng",
        extract_tables: bool = False,
        fixed_layout: Optional[List[TextRegion]] = None,
    ):
        """Creates a PageLayout from an already-loaded PIL Image."""
        page = cls(
            number=number,
            image=image,
            layout=layout,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
            ocr_strategy=ocr_strategy,
            ocr_languages=ocr_languages,
            extract_tables=extract_tables,
        )
        if page.element_extraction_model is not None:
            page.get_elements_using_image_extraction()
            return page
        if fixed_layout is None:
            page.get_elements_with_detection_model()
        else:
            page.elements = page.get_elements_from_layout(fixed_layout)
        return page


def process_data_with_model(
    data: BinaryIO,
    model_name: Optional[str],
    is_image: bool = False,
    ocr_strategy: str = "auto",
    ocr_languages: str = "eng",
    fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
    extract_tables: bool = False,
) -> DocumentLayout:
    """Processes pdf file in the form of a file handler (supporting a read method) into a
    DocumentLayout by using a model identified by model_name."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(data.read())
        layout = process_file_with_model(
            tmp_file.name,
            model_name,
            is_image=is_image,
            ocr_strategy=ocr_strategy,
            ocr_languages=ocr_languages,
            fixed_layouts=fixed_layouts,
            extract_tables=extract_tables,
        )

    return layout


def process_file_with_model(
    filename: str,
    model_name: Optional[str],
    is_image: bool = False,
    ocr_strategy: str = "auto",
    ocr_languages: str = "eng",
    fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
    extract_tables: bool = False,
) -> DocumentLayout:
    """Processes pdf file with name filename into a DocumentLayout by using a model identified by
    model_name."""
    model = get_model(model_name)
    if isinstance(model, UnstructuredObjectDetectionModel):
        detection_model = model
        element_extraction_model = None
    elif isinstance(model, UnstructuredElementExtractionModel):
        detection_model = None
        element_extraction_model = model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    layout = (
        DocumentLayout.from_image_file(
            filename,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
            ocr_strategy=ocr_strategy,
            ocr_languages=ocr_languages,
            extract_tables=extract_tables,
        )
        if is_image
        else DocumentLayout.from_file(
            filename,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
            ocr_strategy=ocr_strategy,
            ocr_languages=ocr_languages,
            fixed_layouts=fixed_layouts,
            extract_tables=extract_tables,
        )
    )
    return layout


def get_element_from_block(
    block: TextRegion,
    image: Optional[Image.Image] = None,
    pdf_objects: Optional[List[TextRegion]] = None,
    ocr_strategy: str = "auto",
    ocr_languages: str = "eng",
    extract_tables: bool = False,
) -> LayoutElement:
    """Creates a LayoutElement from a given layout or image by finding all the text that lies within
    a given block."""
    element = block if isinstance(block, LayoutElement) else LayoutElement.from_region(block)
    element.text = element.extract_text(
        objects=pdf_objects,
        image=image,
        extract_tables=extract_tables,
        ocr_strategy=ocr_strategy,
        ocr_languages=ocr_languages,
    )
    return element


def load_pdf(
    filename: str,
    dpi: int = 200,
    chunk_size: int = 100,
) -> Tuple[List[List[TextRegion]], List[Image.Image]]:
    """Loads the image and word objects from a pdf using pdfplumber and the image renderings of the
    pdf pages using pdf2image"""
    layouts = []
    for page in extract_pages(filename):
        layout = []
        height = page.height
        for element in page:
            x1, y2, x2, y1 = element.bbox
            y1 = height - y1
            y2 = height - y2
            # Coefficient to rescale bounding box to be compatible with images
            coef = dpi / 72
            _text, element_class = (
                (element.get_text(), EmbeddedTextRegion)
                if hasattr(element, "get_text")
                else (None, ImageTextRegion)
            )
            text_region = element_class(x1 * coef, y1 * coef, x2 * coef, y2 * coef, text=_text)

            if text_region.area() > 0:
                layout.append(text_region)
        layouts.append(layout)

    # Convert a PDF in small chunks of pages at a time (e.g. 1-10, 11-20... and so on)
    info = pdf2image.pdfinfo_from_path(filename)
    total_pages = info["Pages"]
    images = []

    for start_page in range(1, total_pages + 1, chunk_size):
        end_page = min(start_page + chunk_size - 1, total_pages)
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_images = pdf2image.convert_from_path(
                filename,
                dpi=dpi,
                first_page=start_page,
                last_page=end_page,
                output_folder=tmpdir,
            )
            images += chunk_images
    return layouts, images
