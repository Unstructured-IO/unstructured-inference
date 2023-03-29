from __future__ import annotations
import os
import re
import tempfile
from tqdm import tqdm
from typing import List, Optional, Tuple, Union, BinaryIO
import unicodedata

import numpy as np
import pdfplumber
import pdf2image
from PIL import Image

from unstructured_inference.inference.elements import TextRegion, ImageTextRegion, LayoutElement
from unstructured_inference.logger import logger
import unstructured_inference.models.tesseract as tesseract
import unstructured_inference.models.tables as tables
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
        image: Image,
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

    def get_elements(self, inplace=True) -> Optional[List[LayoutElement]]:
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
        elements = []
        for e in tqdm(layout):
            elements.append(
                get_element_from_block(
                    e, self.image, self.layout, self.ocr_strategy, self.extract_tables
                )
            )
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
            page.get_elements()
        else:
            page.elements = page.get_elements_from_layout(fixed_layout)
        return page


def process_data_with_model(
    data: BinaryIO,
    model_name: Optional[str],
    is_image: bool = False,
    ocr_strategy: str = "auto",
    fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
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
        )

    return layout


def process_file_with_model(
    filename: str,
    model_name: Optional[str],
    is_image: bool = False,
    ocr_strategy: str = "auto",
    fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
) -> DocumentLayout:
    """Processes pdf file with name filename into a DocumentLayout by using a model identified by
    model_name."""
    model = get_model(model_name)
    layout = (
        DocumentLayout.from_image_file(filename, model=model, ocr_strategy=ocr_strategy)
        if is_image
        else DocumentLayout.from_file(
            filename, model=model, ocr_strategy=ocr_strategy, fixed_layouts=fixed_layouts
        )
    )
    return layout


def cid_ratio(text: str) -> float:
    """Gets ratio of unknown 'cid' characters extracted from text to all characters."""
    if not is_cid_present(text):
        return 0.0
    cid_pattern = r"\(cid\:(\d+)\)"
    unmatched, n_cid = re.subn(cid_pattern, "", text)
    total = n_cid + len(unmatched)
    return n_cid / total


def is_cid_present(text: str) -> bool:
    """Checks if a cid code is present in a text selection."""
    if len(text) < len("(cid:x)"):
        return False
    return text.find("(cid:") != -1


def get_element_from_block(
    block: TextRegion,
    image: Optional[Image.Image] = None,
    pdf_objects: Optional[List[Union[TextRegion, ImageTextRegion]]] = None,
    ocr_strategy: str = "auto",
    extract_tables: bool = False,
) -> LayoutElement:
    """Creates a LayoutElement from a given layout or image by finding all the text that lies within
    a given block."""
    if block.text is not None:
        # If block text is already populated, we'll assume it's correct
        text = block.text
    elif extract_tables and isinstance(block, LayoutElement) and block.type == "Table":
        text = interprete_table_block(block, image)
    elif pdf_objects is not None:
        text = aggregate_by_block(block, image, pdf_objects, ocr_strategy)
    elif image is not None:
        # We don't have anything to go on but the image itself, so we use OCR
        text = ocr(block, image)
    else:
        raise ValueError(
            "Got arguments image and layout as None, at least one must be populated to use for "
            "text extraction."
        )
    element = LayoutElement.from_region(block)
    element.text = text
    return element


def aggregate_by_block(
    text_region: TextRegion,
    image: Optional[Image.Image],
    pdf_objects: List[Union[TextRegion, ImageTextRegion]],
    ocr_strategy: str = "auto",
) -> str:
    """Extracts the text aggregated from the elements of the given layout that lie within the given
    block."""
    word_objects = [obj for obj in pdf_objects if isinstance(obj, TextRegion)]
    image_objects = [obj for obj in pdf_objects if isinstance(obj, ImageTextRegion)]
    if image is not None and needs_ocr(text_region, word_objects, image_objects, ocr_strategy):
        text = ocr(text_region, image)
    else:
        filtered_blocks = [obj for obj in pdf_objects if obj.is_in(text_region, error_margin=5)]
        for little_block in filtered_blocks:
            if image is not None and needs_ocr(
                little_block, word_objects, image_objects, ocr_strategy
            ):
                little_block.text = ocr(little_block, image)
        text = " ".join([x.text for x in filtered_blocks if x.text])
    text = remove_control_characters(text)
    return text


def interprete_table_block(text_block: TextRegion, image: Image.Image) -> str:
    """Extract the contents of a table."""
    tables.load_agent()
    if tables.tables_agent is None:
        raise RuntimeError("Unable to load table extraction agent.")
    padded_block = text_block.pad(12)
    cropped_image = image.crop((padded_block.x1, padded_block.y1, padded_block.x2, padded_block.y2))
    return tables.tables_agent.predict(cropped_image)


def ocr(text_block: TextRegion, image: Image.Image) -> str:
    """Runs a cropped text block image through and OCR agent."""
    logger.debug("Running OCR on text block ...")
    tesseract.load_agent()
    padded_block = text_block.pad(12)
    cropped_image = image.crop((padded_block.x1, padded_block.y1, padded_block.x2, padded_block.y2))
    return tesseract.ocr_agent.detect(cropped_image)


def remove_control_characters(text: str) -> str:
    """Removes control characters from text."""
    out_text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
    return out_text


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
        word_objs = [
            TextRegion(
                x1=word["x0"], y1=word["top"], x2=word["x1"], y2=word["bottom"], text=word["text"]
            )
            for word in plumber_words
        ]
        image_objs = [
            ImageTextRegion(x1=image["x0"], y1=image["y0"], x2=image["x1"], y2=image["y1"])
            for image in page.images
        ]
        layout = word_objs + image_objs
        layouts.append(layout)

    images = pdf2image.convert_from_path(filename, dpi=dpi)
    return layouts, images


def needs_ocr(
    region: TextRegion,
    word_objects: List[TextRegion],
    image_objects: List[ImageTextRegion],
    ocr_strategy: str,
) -> bool:
    """Logic to determine whether ocr is needed to extract text from given region."""
    if ocr_strategy == "force":
        return True
    elif ocr_strategy == "auto":
        # If any image object overlaps with the region of interest, we have hope of getting some
        # text from OCR. Otherwise, there's nothing there to find, no need to waste our time with
        # OCR.
        image_intersects = any(region.intersects(img_obj) for img_obj in image_objects)
        if region.text is None:
            # If the region has no text check if any images overlap with the region that might
            # contain text.
            if any(obj.is_in(region) and obj.text is not None for obj in word_objects):
                # If there are word objects in the region, we defer to that rather than OCR
                return False
            else:
                return image_intersects
        elif cid_ratio(region.text) > 0.5:
            # If the region has text, we should only have to OCR if too much of the text is
            # uninterpretable.
            return image_intersects
        else:
            return False
    else:
        return False
