from __future__ import annotations
import os
import tempfile
from typing import List, Optional, Tuple, Union, BinaryIO

import numpy as np
import pdfplumber
import pdf2image
from PIL import Image

from unstructured_inference.inference.elements import (
    TextRegion,
    EmbeddedTextRegion,
    ImageTextRegion,
)
from unstructured_inference.inference.layoutelement import LayoutElement
from unstructured_inference.logger import logger
from unstructured_inference.models.base import get_model
from unstructured_inference.models.unstructuredmodel import UnstructuredModel

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
        model: Optional[UnstructuredModel] = None,
        fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
        ocr_strategy: str = "auto",
        extract_tables: bool = False,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from a pdf file."""
        logger.info(f"Reading PDF for file: {filename} ...")
        layouts, images = load_pdf(filename)
        if len(layouts) > len(images):
            raise RuntimeError(
                "Some images were not loaded. Check that poppler is installed and in your $PATH."
            )
        pages: List[PageLayout] = list()
        if fixed_layouts is None:
            fixed_layouts = [None for _ in layouts]
        for image, layout, fixed_layout in zip(images, layouts, fixed_layouts):
            # NOTE(robinson) - In the future, maybe we detect the page number and default
            # to the index if it is not detected
            page = PageLayout.from_image(
                image,
                model=model,
                layout=layout,
                ocr_strategy=ocr_strategy,
                fixed_layout=fixed_layout,
                extract_tables=extract_tables,
            )
            pages.append(page)
        return cls.from_pages(pages)

    @classmethod
    def from_image_file(
        cls,
        filename: str,
        model: Optional[UnstructuredModel] = None,
        ocr_strategy: str = "auto",
        fixed_layout: Optional[List[TextRegion]] = None,
        extract_tables: bool = False,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from an image file."""
        logger.info(f"Reading image file: {filename} ...")
        try:
            image = Image.open(filename).convert("RGB")
        except Exception as e:
            if os.path.isdir(filename) or os.path.isfile(filename):
                raise e
            else:
                raise FileNotFoundError(f'File "{filename}" not found!') from e
        page = PageLayout.from_image(
            image,
            model=model,
            layout=None,
            ocr_strategy=ocr_strategy,
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
        model: Optional[UnstructuredModel] = None,
        ocr_strategy: str = "auto",
        extract_tables: bool = False,
    ):
        self.image = image
        self.image_array: Union[np.ndarray, None] = None
        self.layout = layout
        self.number = number
        self.model = model
        self.elements: List[LayoutElement] = list()
        if ocr_strategy not in VALID_OCR_STRATEGIES:
            raise ValueError(f"ocr_strategy must be one of {VALID_OCR_STRATEGIES}.")
        self.ocr_strategy = ocr_strategy
        self.extract_tables = extract_tables

    def __str__(self) -> str:
        return "\n\n".join([str(element) for element in self.elements])

    def get_elements_with_model(self, inplace=True) -> Optional[List[LayoutElement]]:
        """Uses specified model to detect the elements on the page."""
        logger.info("Detecting page elements ...")
        if self.model is None:
            self.model = get_model()

        # NOTE(mrobinson) - We'll want make this model inference step some kind of
        # remote call in the future.
        inferred_layout = self.model(self.image)
        elements = self.get_elements_from_layout(inferred_layout)
        if inplace:
            self.elements = elements
            return None
        return elements

    def get_elements_from_layout(self, layout: List[TextRegion]) -> List[LayoutElement]:
        """Uses the given Layout to separate the page text into elements, either extracting the
        text from the discovered layout blocks or from the image using OCR."""
        # NOTE(robinson) - This orders the page from top to bottom. We'll need more
        # sophisticated ordering logic for more complicated layouts.
        layout.sort(key=lambda element: element.y1)
        elements = [
            get_element_from_block(
                e, self.image, self.layout, self.ocr_strategy, self.extract_tables
            )
            for e in layout
        ]
        return elements

    def _get_image_array(self) -> Union[np.ndarray, None]:
        """Converts the raw image into a numpy array."""
        if self.image_array is None:
            self.image_array = np.array(self.image)
        return self.image_array

    @classmethod
    def from_image(
        cls,
        image,
        model: Optional[UnstructuredModel] = None,
        layout: Optional[List[TextRegion]] = None,
        ocr_strategy: str = "auto",
        extract_tables: bool = False,
        fixed_layout: Optional[List[TextRegion]] = None,
    ):
        """Creates a PageLayout from an already-loaded PIL Image."""
        page = cls(
            number=0,
            image=image,
            layout=layout,
            model=model,
            ocr_strategy=ocr_strategy,
            extract_tables=extract_tables,
        )
        if fixed_layout is None:
            page.get_elements_with_model()
        else:
            page.elements = page.get_elements_from_layout(fixed_layout)
        return page


def process_data_with_model(
    data: BinaryIO,
    model_name: Optional[str],
    is_image: bool = False,
    ocr_strategy: str = "auto",
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
            fixed_layouts=fixed_layouts,
            extract_tables=extract_tables,
        )

    return layout


def process_file_with_model(
    filename: str,
    model_name: Optional[str],
    is_image: bool = False,
    ocr_strategy: str = "auto",
    fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
    extract_tables: bool = False,
) -> DocumentLayout:
    """Processes pdf file with name filename into a DocumentLayout by using a model identified by
    model_name."""
    model = get_model(model_name)
    layout = (
        DocumentLayout.from_image_file(
            filename, model=model, ocr_strategy=ocr_strategy, extract_tables=extract_tables
        )
        if is_image
        else DocumentLayout.from_file(
            filename,
            model=model,
            ocr_strategy=ocr_strategy,
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
    extract_tables: bool = False,
) -> LayoutElement:
    """Creates a LayoutElement from a given layout or image by finding all the text that lies within
    a given block."""
    element = LayoutElement.from_region(block)
    element.text = block.extract_text(
        objects=pdf_objects, image=image, extract_tables=extract_tables, ocr_strategy=ocr_strategy
    )
    return element


def load_pdf(
    filename: str,
    x_tolerance: Union[int, float] = 1.5,
    y_tolerance: Union[int, float] = 2,
    keep_blank_chars: bool = False,
    use_text_flow: bool = False,
    horizontal_ltr: bool = True,  # Should words be read left-to-right?
    vertical_ttb: bool = True,  # Should vertical words be read top-to-bottom?
    extra_attrs: Optional[List[str]] = None,
    split_at_punctuation: Union[bool, str] = False,
    dpi: int = 200,
) -> Tuple[List[List[TextRegion]], List[Image.Image]]:
    """Loads the image and word objects from a pdf using pdfplumber and the image renderings of the
    pdf pages using pdf2image"""
    pdf_object = pdfplumber.open(filename)
    layouts = []
    images = []
    for page in pdf_object.pages:
        plumber_words = page.extract_words(
            x_tolerance=x_tolerance,
            y_tolerance=y_tolerance,
            keep_blank_chars=keep_blank_chars,
            use_text_flow=use_text_flow,
            horizontal_ltr=horizontal_ltr,
            vertical_ttb=vertical_ttb,
            extra_attrs=extra_attrs,
            split_at_punctuation=split_at_punctuation,
        )
        word_objs: List[TextRegion] = [
            EmbeddedTextRegion(
                x1=word["x0"] * dpi / 72,
                y1=word["top"] * dpi / 72,
                x2=word["x1"] * dpi / 72,
                y2=word["bottom"] * dpi / 72,
                text=word["text"],
            )
            for word in plumber_words
        ]
        image_objs: List[TextRegion] = [
            ImageTextRegion(
                x1=image["x0"] * dpi / 72,
                y1=image["top"] * dpi / 72,
                x2=image["x1"] * dpi / 72,
                y2=image["bottom"] * dpi / 72,
            )
            for image in page.images
        ]
        layout = word_objs + image_objs
        layouts.append(layout)

    images = pdf2image.convert_from_path(filename, dpi=dpi)
    return layouts, images
