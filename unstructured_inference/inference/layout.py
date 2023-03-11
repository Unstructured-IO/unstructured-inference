from __future__ import annotations
from dataclasses import dataclass
import os
import re
import tempfile
from tqdm import tqdm
from typing import List, Optional, Tuple, Union, BinaryIO
import unicodedata
from layoutparser.io.pdf import load_pdf
from layoutparser.elements.layout_elements import TextBlock
from layoutparser.elements.layout import Layout
import numpy as np
from PIL import Image

from unstructured_inference.logger import logger
import unstructured_inference.models.tesseract as tesseract
from unstructured_inference.models.base import get_model
from unstructured_inference.models.unstructuredmodel import UnstructuredModel

VALID_OCR_STRATEGIES = (
    "auto",  # Use OCR when it looks like other methods have failed
    "force",  # Always use OCR
    "never",  # Never use OCR
)


@dataclass
# NOTE(alan): I notice this has (almost?) the same structure as a layoutparser TextBlock. Maybe we
# don't need to make our own here?
class LayoutElement:
    type: str
    # NOTE(robinson) - The list contain two elements, each a tuple
    # in format (x1,y1), the first the upper left corner and the second
    # the right bottom corner
    coordinates: List[Tuple[float, float]]
    text: Optional[str] = None

    def __str__(self) -> str:
        return str(self.text)

    def to_dict(self) -> dict:
        return self.__dict__

    def get_width(self) -> float:
        # NOTE(benjamin) i.e: y2-y1
        return self.coordinates[1][0] - self.coordinates[0][0]

    def get_height(self) -> float:
        # NOTE(benjamin) i.e: x2-x1
        return self.coordinates[1][1] - self.coordinates[0][1]


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
        fixed_layouts: Optional[List[Optional[Layout]]] = None,
        ocr_strategy: str = "auto",
    ) -> DocumentLayout:
        """Creates a DocumentLayout from a pdf file."""
        # NOTE(alan): For now the model is a Detectron2LayoutModel but in the future it should
        # be an abstract class that supports some standard interface and can accomodate either
        # a locally instantiated model or an API. Maybe even just a callable that accepts an
        # image and returns a dict, or something.
        logger.info(f"Reading PDF for file: {filename} ...")
        layouts, images = load_pdf(filename, load_images=True)
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
            )
            pages.append(page)
        return cls.from_pages(pages)

    @classmethod
    def from_image_file(
        cls,
        filename: str,
        model: Optional[UnstructuredModel] = None,
        ocr_strategy: str = "auto",
        fixed_layout: Optional[Layout] = None,
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
            image, model=model, layout=None, ocr_strategy=ocr_strategy, fixed_layout=fixed_layout
        )
        return cls.from_pages([page])


class PageLayout:
    """Class for an individual PDF page."""

    def __init__(
        self,
        number: int,
        image: Image,
        layout: Layout,
        model: Optional[UnstructuredModel] = None,
        ocr_strategy: str = "auto",
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

    def get_elements_from_layout(self, layout: Layout) -> List[LayoutElement]:
        """Uses the given Layout to separate the page text into elements, either extracting the
        text from the discovered layout blocks or from the image using OCR."""
        # NOTE(robinson) - This orders the page from top to bottom. We'll need more
        # sophisticated ordering logic for more complicated layouts.
        layout.sort(key=lambda element: element.coordinates[1], inplace=True)
        # NOTE(benjamin): Creates a Pool for concurrent processing of image elements by OCR
        elements = []
        for e in tqdm(layout):
            elements.append(get_element_from_block(e, self.image, self.layout, self.ocr_strategy))
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
        layout: Optional[Layout] = None,
        ocr_strategy: str = "auto",
        fixed_layout: Optional[Layout] = None,
    ):
        """Creates a PageLayout from an already-loaded PIL Image."""
        page = cls(number=0, image=image, layout=layout, model=model, ocr_strategy=ocr_strategy)
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
    fixed_layouts: Optional[List[Optional[Layout]]] = None,
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
    fixed_layouts: Optional[List[Optional[Layout]]] = None,
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
    block: TextBlock,
    image: Optional[Image.Image] = None,
    layout: Optional[Layout] = None,
    ocr_strategy: str = "auto",
) -> LayoutElement:
    """Creates a LayoutElement from a given layout or image by finding all the text that lies within
    a given block."""
    if block.text is not None:
        # If block text is already populated, we'll assume it's correct
        text = block.text
    elif layout is not None:
        text = aggregate_by_block(block, image, layout, ocr_strategy)
    elif image is not None:
        text = interpret_text_block(block, image, ocr_strategy)
    else:
        raise ValueError(
            "Got arguments image and layout as None, at least one must be populated to use for "
            "text extraction."
        )
    element = LayoutElement(type=block.type, text=text, coordinates=block.points.tolist())
    return element


def aggregate_by_block(
    text_block: TextBlock,
    image: Optional[Image.Image],
    layout: Layout,
    ocr_strategy: str = "auto",
) -> str:
    """Extracts the text aggregated from the elements of the given layout that lie within the given
    block."""
    filtered_blocks = layout.filter_by(text_block, center=True)
    # NOTE(alan): For now, if none of the elements discovered by layoutparser are in the block
    # we can try interpreting the whole block. This still doesn't handle edge cases, like when there
    # are some text elements within the block, but there are image elements overlapping the block
    # with text lying within the block. In this case the text in the image would likely be ignored.
    if not filtered_blocks:
        text = interpret_text_block(text_block, image, ocr_strategy)
        return text
    for little_block in filtered_blocks:
        little_block.text = interpret_text_block(little_block, image, ocr_strategy)
    text = " ".join([x for x in filtered_blocks.get_texts() if x])
    return text


def interpret_text_block(
    text_block: TextBlock, image: Image.Image, ocr_strategy: str = "auto"
) -> str:
    """Interprets the text in a TextBlock using OCR or the text attribute, according to the given
    ocr_strategy."""
    # NOTE(robinson) - If the text attribute is None, that means the PDF isn't
    # already OCR'd and we have to send the snippet out for OCRing.

    if (ocr_strategy == "force") or (
        ocr_strategy == "auto" and ((text_block.text is None) or cid_ratio(text_block.text) > 0.5)
    ):
        out_text = ocr(text_block, image)
    else:
        out_text = "" if text_block.text is None else text_block.text
    out_text = remove_control_characters(out_text)
    return out_text


def ocr(text_block: TextBlock, image: Image.Image) -> str:
    """Runs a cropped text block image through and OCR agent."""
    logger.debug("Running OCR on text block ...")
    tesseract.load_agent()
    image_array = np.array(image)
    padded_block = text_block.pad(left=5, right=5, top=5, bottom=5)
    cropped_image = padded_block.crop_image(image_array)
    return tesseract.ocr_agent.detect(cropped_image)


def remove_control_characters(text: str) -> str:
    """Removes control characters from text."""
    out_text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
    return out_text
