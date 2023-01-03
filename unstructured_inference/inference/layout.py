from __future__ import annotations
from dataclasses import dataclass
import tempfile
from typing import List, Optional, Tuple, Union, BinaryIO

import layoutparser as lp
from layoutparser.models.detectron2.layoutmodel import Detectron2LayoutModel
import numpy as np
from PIL import Image

from unstructured_inference.logger import logger
import unstructured_inference.models.tesseract as tesseract
import unstructured_inference.models.detectron2 as detectron2
from unstructured_inference.models import get_model


@dataclass
class LayoutElement:
    type: str
    # NOTE(robinson) - Coordinates are reported starting from the upper left and
    # proceeding clockwise
    coordinates: List[Tuple[float, float]]
    text: Optional[str] = None

    def __str__(self):
        return self.text

    def to_dict(self):
        return self.__dict__


class DocumentLayout:
    """Class for handling documents that are saved as .pdf files. For .pdf files, a
    document image analysis (DIA) model detects the layout of the page prior to extracting
    element."""

    def __init__(self):
        self._pages = None

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
    def from_file(cls, filename: str, model: Optional[Detectron2LayoutModel] = None):
        """Creates a DocumentLayout from a pdf file."""
        # NOTE(alan): For now the model is a Detectron2LayoutModel but in the future it should
        # be an abstract class that supports some standard interface and can accomodate either
        # a locally instantiated model or an API. Maybe even just a callable that accepts an
        # image and returns a dict, or something.
        logger.info(f"Reading PDF for file: {filename} ...")
        layouts, images = lp.load_pdf(filename, load_images=True)
        pages: List[PageLayout] = list()
        for i, layout in enumerate(layouts):
            image = images[i]
            # NOTE(robinson) - In the future, maybe we detect the page number and default
            # to the index if it is not detected
            page = PageLayout(number=i, image=image, layout=layout, model=model)
            page.get_elements()
            pages.append(page)
        return cls.from_pages(pages)


class PageLayout:
    """Class for an individual PDF page."""

    def __init__(
        self,
        number: int,
        image: Image,
        layout: lp.Layout,
        model: Optional[Detectron2LayoutModel] = None,
    ):
        self.image = image
        self.image_array: Union[np.ndarray, None] = None
        self.layout = layout
        self.number = number
        self.model = model
        self.elements: List[LayoutElement] = list()

    def __str__(self):
        return "\n\n".join([str(element) for element in self.elements])

    def get_elements(self, inplace=True) -> Optional[List[LayoutElement]]:
        """Uses a layoutparser model to detect the elements on the page."""
        logger.info("Detecting page elements ...")
        if self.model is None:
            self.model = detectron2.load_default_model()

        elements = list()
        # NOTE(mrobinson) - We'll want make this model inference step some kind of
        # remote call in the future.
        image_layout = self.model.detect(self.image)
        # NOTE(robinson) - This orders the page from top to bottom. We'll need more
        # sophisticated ordering logic for more complicated layouts.
        image_layout.sort(key=lambda element: element.coordinates[1], inplace=True)
        for item in image_layout:
            text_blocks = self.layout.filter_by(item, center=True)
            text = str()
            for text_block in text_blocks:
                # NOTE(robinson) - If the text attribute is None, that means the PDF isn't
                # already OCR'd and we have to send the snippet out for OCRing.
                if text_block.text is None:
                    text_block.text = self.ocr(text_block)
            text = " ".join([x for x in text_blocks.get_texts() if x])

            elements.append(
                LayoutElement(type=item.type, text=text, coordinates=item.points.tolist())
            )

        if inplace:
            self.elements = elements
            return None
        return elements

    def ocr(self, text_block: lp.TextBlock) -> str:
        """Runs a cropped text block image through and OCR agent."""
        logger.debug("Running OCR on text block ...")
        tesseract.load_agent()
        image_array = self._get_image_array()
        padded_block = text_block.pad(left=5, right=5, top=5, bottom=5)
        cropped_image = padded_block.crop_image(image_array)
        return tesseract.ocr_agent.detect(cropped_image)

    def _get_image_array(self) -> Union[np.ndarray, None]:
        """Converts the raw image into a numpy array."""
        if self.image_array is None:
            self.image_array = np.array(self.image)
        return self.image_array


def process_data_with_model(data: BinaryIO, model_name: str) -> DocumentLayout:
    """Processes pdf file in the form of a file handler (supporting a read method) into a
    DocumentLayout by using a model identified by model_name."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(data.read())
        layout = process_file_with_model(tmp_file.name, model_name)

    return layout


def process_file_with_model(filename: str, model_name: str) -> DocumentLayout:
    """Processes pdf file with name filename into a DocumentLayout by using a model identified by
    model_name."""
    model = None if model_name is None else get_model(model_name)
    layout = DocumentLayout.from_file(filename, model=model)
    return layout
