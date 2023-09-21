from __future__ import annotations

import os
import tempfile
from pathlib import PurePath
from typing import BinaryIO, Collection, List, Optional, Tuple, Union, cast

import numpy as np
import pdf2image
import pytesseract
from pdfminer import psparser
from pdfminer.high_level import extract_pages
from PIL import Image, ImageSequence
from pytesseract import Output

from unstructured_inference.constants import OCRMode
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
    merge_inferred_layout_with_ocr_layout,
)
from unstructured_inference.inference.ordering import order_layout
from unstructured_inference.inference.pdf import get_images_from_pdf_element
from unstructured_inference.logger import logger
from unstructured_inference.models.base import get_model
from unstructured_inference.models.detectron2onnx import (
    UnstructuredDetectronONNXModel,
)
from unstructured_inference.models.unstructuredmodel import (
    UnstructuredElementExtractionModel,
    UnstructuredObjectDetectionModel,
)
from unstructured_inference.patches.pdfminer import parse_keyword
from unstructured_inference.utils import write_image
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
        ocr_mode: str = OCRMode.FULL_PAGE.value,
        extract_tables: bool = False,
        pdf_image_dpi: int = 200,
        **kwargs,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from a pdf file."""
        logger.info(f"Reading PDF for file: {filename} ...")

        with tempfile.TemporaryDirectory() as temp_dir:
            layouts, _image_paths = load_pdf(
                filename,
                pdf_image_dpi,
                output_folder=temp_dir,
                path_only=True,
            )
            image_paths = cast(List[str], _image_paths)
            if len(layouts) > len(image_paths):
                raise RuntimeError(
                    "Some images were not loaded. "
                    "Check that poppler is installed and in your $PATH.",
                )

            pages: List[PageLayout] = []
            if fixed_layouts is None:
                fixed_layouts = [None for _ in layouts]
            for i, (image_path, layout, fixed_layout) in enumerate(
                zip(image_paths, layouts, fixed_layouts),
            ):
                # NOTE(robinson) - In the future, maybe we detect the page number and default
                # to the index if it is not detected
                with Image.open(image_path) as image:
                    page = PageLayout.from_image(
                        image,
                        number=i + 1,
                        document_filename=filename,
                        detection_model=detection_model,
                        element_extraction_model=element_extraction_model,
                        layout=layout,
                        ocr_strategy=ocr_strategy,
                        ocr_languages=ocr_languages,
                        ocr_mode=ocr_mode,
                        fixed_layout=fixed_layout,
                        extract_tables=extract_tables,
                        **kwargs,
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
        ocr_mode: str = OCRMode.FULL_PAGE.value,
        fixed_layout: Optional[List[TextRegion]] = None,
        extract_tables: bool = False,
        **kwargs,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from an image file."""
        logger.info(f"Reading image file: {filename} ...")
        try:
            image = Image.open(filename)
            format = image.format
            images = []
            for i, im in enumerate(ImageSequence.Iterator(image)):
                im = im.convert("RGB")
                im.format = format
                images.append(im)
        except Exception as e:
            if os.path.isdir(filename) or os.path.isfile(filename):
                raise e
            else:
                raise FileNotFoundError(f'File "{filename}" not found!') from e
        pages = []
        for i, image in enumerate(images):
            page = PageLayout.from_image(
                image,
                image_path=filename,
                number=i,
                detection_model=detection_model,
                element_extraction_model=element_extraction_model,
                layout=None,
                ocr_strategy=ocr_strategy,
                ocr_languages=ocr_languages,
                ocr_mode=ocr_mode,
                fixed_layout=fixed_layout,
                extract_tables=extract_tables,
                **kwargs,
            )
            pages.append(page)
        return cls.from_pages(pages)


class PageLayout:
    """Class for an individual PDF page."""

    def __init__(
        self,
        number: int,
        image: Image.Image,
        layout: Optional[List[TextRegion]],
        image_metadata: Optional[dict] = None,
        image_path: Optional[Union[str, PurePath]] = None,  # TODO: Deprecate
        document_filename: Optional[Union[str, PurePath]] = None,
        detection_model: Optional[UnstructuredObjectDetectionModel] = None,
        element_extraction_model: Optional[UnstructuredElementExtractionModel] = None,
        ocr_strategy: str = "auto",
        ocr_languages: str = "eng",
        ocr_mode: str = OCRMode.FULL_PAGE.value,
        extract_tables: bool = False,
        analysis: bool = False,
        supplement_with_ocr_elements: bool = True,
    ):
        if detection_model is not None and element_extraction_model is not None:
            raise ValueError("Only one of detection_model and extraction_model should be passed.")
        self.image = image
        if image_metadata is None:
            image_metadata = {}
        self.image_metadata = image_metadata
        self.image_path = image_path
        self.image_array: Union[np.ndarray, None] = None
        self.document_filename = document_filename
        self.layout = layout
        self.number = number
        self.detection_model = detection_model
        self.element_extraction_model = element_extraction_model
        self.elements: Collection[Union[LayoutElement, LocationlessLayoutElement]] = []
        if ocr_strategy not in VALID_OCR_STRATEGIES:
            raise ValueError(f"ocr_strategy must be one of {VALID_OCR_STRATEGIES}.")
        self.ocr_strategy = ocr_strategy
        self.ocr_languages = ocr_languages
        self.ocr_mode = ocr_mode
        self.extract_tables = extract_tables
        self.analysis = analysis
        self.inferred_layout: Optional[List[LayoutElement]] = None
        self.ocr_layout: Optional[List[TextRegion]] = None
        self.supplement_with_ocr_elements = supplement_with_ocr_elements

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

    def get_elements_with_detection_model(
        self,
        inplace: bool = True,
    ) -> Optional[List[LayoutElement]]:
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
        inferred_layout: List[LayoutElement] = self.detection_model(self.image)
        inferred_layout = UnstructuredObjectDetectionModel.deduplicate_detected_elements(
            inferred_layout,
        )
        if self.ocr_mode == OCRMode.INDIVIDUAL_BLOCKS.value:
            ocr_layout = None
        elif self.ocr_mode == OCRMode.FULL_PAGE.value:
            ocr_layout = None
            entrie_page_ocr = os.getenv("ENTIRE_PAGE_OCR", "tesseract").lower()
            if entrie_page_ocr not in ["paddle", "tesseract"]:
                raise ValueError(
                    "Environment variable ENTIRE_PAGE_OCR must be set to 'tesseract' or 'paddle'.",
                )

            if entrie_page_ocr == "paddle":
                logger.info("Processing entrie page OCR with paddle...")
                from unstructured_inference.models import paddle_ocr

                # TODO(yuming): paddle only support one language at once,
                # change ocr to tesseract if passed in multilanguages.
                ocr_data = paddle_ocr.load_agent(language=self.ocr_languages).ocr(
                    np.array(self.image),
                    cls=True,
                )
                ocr_layout = parse_ocr_data_paddle(ocr_data)
            else:
                logger.info("Processing entrie page OCR with tesseract...")
                try:
                    ocr_data = pytesseract.image_to_data(
                        self.image,
                        lang=self.ocr_languages,
                        output_type=Output.DICT,
                    )
                    ocr_layout = parse_ocr_data_tesseract(ocr_data)
                except pytesseract.pytesseract.TesseractError:
                    logger.warning("TesseractError: Skipping page", exc_info=True)

        if self.layout is not None:
            threshold_kwargs = {}
            # NOTE(Benjamin): With this the thresholds are only changed for detextron2_mask_rcnn
            # In other case the default values for the functions are used
            if (
                isinstance(self.detection_model, UnstructuredDetectronONNXModel)
                and "R_50" not in self.detection_model.model_path
            ):
                threshold_kwargs = {"same_region_threshold": 0.5, "subregion_threshold": 0.5}
            merged_layout = merge_inferred_layout_with_extracted_layout(
                inferred_layout=inferred_layout,
                extracted_layout=self.layout,
                page_image_size=self.image.size,
                ocr_layout=ocr_layout,
                supplement_with_ocr_elements=self.supplement_with_ocr_elements,
                **threshold_kwargs,
            )
        elif ocr_layout is not None:
            threshold_kwargs = {}
            # NOTE(Benjamin): With this the thresholds are only changed for detextron2_mask_rcnn
            # In other case the default values for the functions are used
            if (
                isinstance(self.detection_model, UnstructuredDetectronONNXModel)
                and "R_50" not in self.detection_model.model_path
            ):
                threshold_kwargs = {"subregion_threshold": 0.3}
            merged_layout = merge_inferred_layout_with_ocr_layout(
                inferred_layout=inferred_layout,
                ocr_layout=ocr_layout,
                supplement_with_ocr_elements=self.supplement_with_ocr_elements,
                **threshold_kwargs,
            )
        else:
            merged_layout = inferred_layout

        elements = self.get_elements_from_layout(cast(List[TextRegion], merged_layout))

        if self.analysis:
            self.inferred_layout = inferred_layout
            self.ocr_layout = ocr_layout

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

    def extract_images(self, output_dir_path: Optional[str] = None):
        """
        Extract and save images from the page. This method iterates through the layout elements
        of the page, identifies image regions, and extracts and saves them as separate image files.
        """

        if not output_dir_path:
            output_dir_path = os.path.join(os.getcwd(), "figures")
        os.makedirs(output_dir_path, exist_ok=True)

        figure_number = 0
        for el in self.elements:
            if isinstance(el, LocationlessLayoutElement) or el.type not in ["Image"]:
                continue

            figure_number += 1
            try:
                output_f_path = os.path.join(
                    output_dir_path,
                    f"figure-{self.number}-{figure_number}.jpg",
                )
                cropped_image = self.image.crop((el.x1, el.y1, el.x2, el.y2))
                write_image(cropped_image, output_f_path)
                el.image_path = output_f_path
            except (ValueError, IOError):
                logger.warning("Image Extraction Error: Skipping the failed image", exc_info=True)

    def _get_image_array(self) -> Union[np.ndarray, None]:
        """Converts the raw image into a numpy array."""
        if self.image_array is None:
            if self.image:
                self.image_array = np.array(self.image)
            else:
                image = Image.open(self.image_path)
                self.image_array = np.array(image)
        return self.image_array

    def annotate(
        self,
        colors: Optional[Union[List[str], str]] = None,
        image_dpi: int = 200,
        annotation_data: Optional[dict[str, dict]] = None,
        add_details: bool = False,
        sources: List[str] = ["all"],
    ) -> Image.Image:
        """Annotates the elements on the page image.
        if add_details is True, and the elements contain type and source attributes, then
        the type and source will be added to the image.
        sources is a list of sources to annotate. If sources is ["all"], then all sources will be
        annotated. Current sources allowed are "yolox","detectron2_onnx" and "detectron2_lp" """
        if colors is None:
            colors = ["red" for _ in self.elements]
        if isinstance(colors, str):
            colors = [colors]
        # If there aren't enough colors, just cycle through the colors a few times
        if len(colors) < len(self.elements):
            n_copies = (len(self.elements) // len(colors)) + 1
            colors = colors * n_copies

        # Hotload image if it hasn't been loaded yet
        if self.image:
            img = self.image.copy()
        elif self.image_path:
            img = Image.open(self.image_path)
        else:
            img = self._get_image(self.document_filename, self.number, image_dpi)

        if annotation_data is None:
            for el, color in zip(self.elements, colors):
                if isinstance(el, Rectangle):
                    required_source = getattr(el, "source", None)
                    if "all" in sources or required_source in sources:
                        img = draw_bbox(img, el, color=color, details=add_details)
        else:
            for attribute, style in annotation_data.items():
                if hasattr(self, attribute) and getattr(self, attribute):
                    color = style["color"]
                    width = style["width"]
                    for region in getattr(self, attribute):
                        if isinstance(region, Rectangle):
                            required_source = getattr(el, "source", None)
                            if "all" in sources or required_source in sources:
                                img = draw_bbox(
                                    img,
                                    region,
                                    color=color,
                                    width=width,
                                    details=add_details,
                                )

        return img

    def _get_image(self, filename, page_number, pdf_image_dpi: int = 200) -> Image.Image:
        """Hotloads a page image from a pdf file."""

        with tempfile.TemporaryDirectory() as temp_dir:
            _image_paths = pdf2image.convert_from_path(
                filename,
                dpi=pdf_image_dpi,
                output_folder=temp_dir,
                paths_only=True,
            )
            image_paths = cast(List[str], _image_paths)
            if page_number > len(image_paths):
                raise ValueError(
                    f"Page number {page_number} is greater than the number of pages in the PDF.",
                )

            with Image.open(image_paths[page_number - 1]) as image:
                return image.copy()

    @classmethod
    def from_image(
        cls,
        image: Image.Image,
        image_path: Optional[Union[str, PurePath]] = None,
        document_filename: Optional[Union[str, PurePath]] = None,
        number: int = 1,
        detection_model: Optional[UnstructuredObjectDetectionModel] = None,
        element_extraction_model: Optional[UnstructuredElementExtractionModel] = None,
        layout: Optional[List[TextRegion]] = None,
        ocr_strategy: str = "auto",
        ocr_languages: str = "eng",
        ocr_mode: str = OCRMode.FULL_PAGE.value,
        extract_tables: bool = False,
        fixed_layout: Optional[List[TextRegion]] = None,
        supplement_with_ocr_elements: bool = True,
        extract_images_in_pdf: bool = False,
        image_output_dir_path: Optional[str] = None,
        analysis: bool = False,
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
            ocr_mode=ocr_mode,
            extract_tables=extract_tables,
            analysis=analysis,
            supplement_with_ocr_elements=supplement_with_ocr_elements,
        )
        if page.element_extraction_model is not None:
            page.get_elements_using_image_extraction()
            return page
        if fixed_layout is None:
            page.get_elements_with_detection_model()
        else:
            page.elements = page.get_elements_from_layout(fixed_layout)

        page.image_metadata = {
            "format": page.image.format if page.image else None,
            "width": page.image.width if page.image else None,
            "height": page.image.height if page.image else None,
        }
        page.image_path = os.path.abspath(image_path) if image_path else None
        page.document_filename = os.path.abspath(document_filename) if document_filename else None

        if extract_images_in_pdf:
            page.extract_images(image_output_dir_path)

        # Clear the image to save memory
        page.image = None

        return page


def process_data_with_model(
    data: BinaryIO,
    model_name: Optional[str],
    is_image: bool = False,
    ocr_strategy: str = "auto",
    ocr_languages: str = "eng",
    ocr_mode: str = OCRMode.FULL_PAGE.value,
    fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
    extract_tables: bool = False,
    pdf_image_dpi: Optional[int] = None,
    **kwargs,
) -> DocumentLayout:
    """Processes pdf file in the form of a file handler (supporting a read method) into a
    DocumentLayout by using a model identified by model_name."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(data.read())
        tmp_file.flush()  # Make sure the file is written out
        layout = process_file_with_model(
            tmp_file.name,
            model_name,
            is_image=is_image,
            ocr_strategy=ocr_strategy,
            ocr_languages=ocr_languages,
            ocr_mode=ocr_mode,
            fixed_layouts=fixed_layouts,
            extract_tables=extract_tables,
            pdf_image_dpi=pdf_image_dpi,
            **kwargs,
        )

    return layout


def process_file_with_model(
    filename: str,
    model_name: Optional[str],
    is_image: bool = False,
    ocr_strategy: str = "auto",
    ocr_languages: str = "eng",
    ocr_mode: str = OCRMode.FULL_PAGE.value,
    fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
    extract_tables: bool = False,
    pdf_image_dpi: Optional[int] = None,
    **kwargs,
) -> DocumentLayout:
    """Processes pdf file with name filename into a DocumentLayout by using a model identified by
    model_name."""

    if pdf_image_dpi is None:
        pdf_image_dpi = 300 if model_name == "chipper" else 200
    if (pdf_image_dpi < 300) and (model_name == "chipper"):
        logger.warning(
            "The Chipper model performs better when images are rendered with DPI >= 300 "
            f"(currently {pdf_image_dpi}).",
        )

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
            ocr_mode=ocr_mode,
            extract_tables=extract_tables,
            **kwargs,
        )
        if is_image
        else DocumentLayout.from_file(
            filename,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
            ocr_strategy=ocr_strategy,
            ocr_languages=ocr_languages,
            ocr_mode=ocr_mode,
            fixed_layouts=fixed_layouts,
            extract_tables=extract_tables,
            pdf_image_dpi=pdf_image_dpi,
            **kwargs,
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
    output_folder: Optional[Union[str, PurePath]] = None,
    path_only: bool = False,
) -> Tuple[List[List[TextRegion]], Union[List[Image.Image], List[str]]]:
    """Loads the image and word objects from a pdf using pdfplumber and the image renderings of the
    pdf pages using pdf2image"""

    layouts = []
    for page in extract_pages(filename):
        layout: List[TextRegion] = []
        height = page.height
        for element in page:
            x1, y2, x2, y1 = element.bbox
            y1 = height - y1
            y2 = height - y2
            # Coefficient to rescale bounding box to be compatible with images
            coef = dpi / 72

            if hasattr(element, "get_text"):
                _text = element.get_text()
                element_class = EmbeddedTextRegion  # type: ignore
            else:
                embedded_images = get_images_from_pdf_element(element)
                if len(embedded_images) > 0:
                    _text = None
                    element_class = ImageTextRegion  # type: ignore
                else:
                    continue

            text_region = element_class(x1 * coef, y1 * coef, x2 * coef, y2 * coef, text=_text)

            if text_region.area > 0:
                layout.append(text_region)
        layouts.append(layout)

    if path_only and not output_folder:
        raise ValueError("output_folder must be specified if path_only is true")

    if output_folder is not None:
        images = pdf2image.convert_from_path(
            filename,
            dpi=dpi,
            output_folder=output_folder,
            paths_only=path_only,
        )
    else:
        images = pdf2image.convert_from_path(
            filename,
            dpi=dpi,
            paths_only=path_only,
        )

    return layouts, images


def parse_ocr_data_tesseract(ocr_data: dict) -> List[TextRegion]:
    """
    Parse the OCR result data to extract a list of TextRegion objects from
    tesseract.

    The function processes the OCR result dictionary, looking for bounding
    box information and associated text to create instances of the TextRegion
    class, which are then appended to a list.

    Parameters:
    - ocr_data (dict): A dictionary containing the OCR result data, expected
                      to have keys like "level", "left", "top", "width",
                      "height", and "text".

    Returns:
    - List[TextRegion]: A list of TextRegion objects, each representing a
                        detected text region within the OCR-ed image.

    Note:
    - An empty string or a None value for the 'text' key in the input
      dictionary will result in its associated bounding box being ignored.
    """

    levels = ocr_data["level"]
    text_regions = []
    for i, level in enumerate(levels):
        (l, t, w, h) = (
            ocr_data["left"][i],
            ocr_data["top"][i],
            ocr_data["width"][i],
            ocr_data["height"][i],
        )
        (x1, y1, x2, y2) = l, t, l + w, t + h
        text = ocr_data["text"][i]
        if text:
            text_region = TextRegion(x1, y1, x2, y2, text=text, source="OCR")
            text_regions.append(text_region)

    return text_regions


def parse_ocr_data_paddle(ocr_data: list) -> List[TextRegion]:
    """
    Parse the OCR result data to extract a list of TextRegion objects from
    paddle.

    The function processes the OCR result dictionary, looking for bounding
    box information and associated text to create instances of the TextRegion
    class, which are then appended to a list.

    Parameters:
    - ocr_data (list): A list containing the OCR result data

    Returns:
    - List[TextRegion]: A list of TextRegion objects, each representing a
                        detected text region within the OCR-ed image.

    Note:
    - An empty string or a None value for the 'text' key in the input
      dictionary will result in its associated bounding box being ignored.
    """
    text_regions = []
    for idx in range(len(ocr_data)):
        res = ocr_data[idx]
        for line in res:
            x1 = min([i[0] for i in line[0]])
            y1 = min([i[1] for i in line[0]])
            x2 = max([i[0] for i in line[0]])
            y2 = max([i[1] for i in line[0]])
            text = line[1][0]
            if text:
                text_region = TextRegion(x1, y1, x2, y2, text)
                text_regions.append(text_region)

    return text_regions
