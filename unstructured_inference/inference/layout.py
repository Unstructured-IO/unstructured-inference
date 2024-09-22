from __future__ import annotations

import os
import tempfile
from pathlib import PurePath
from typing import Any, BinaryIO, Collection, List, Optional, Union, cast

import numpy as np
import pdf2image
from PIL import Image, ImageSequence

from unstructured_inference.inference.elements import (
    TextRegion,
)
from unstructured_inference.inference.layoutelement import LayoutElement, LayoutElements
from unstructured_inference.logger import logger
from unstructured_inference.models.base import get_model
from unstructured_inference.models.unstructuredmodel import (
    UnstructuredElementExtractionModel,
    UnstructuredObjectDetectionModel,
)
from unstructured_inference.visualize import draw_bbox


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
        fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
        pdf_image_dpi: int = 200,
        **kwargs,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from a pdf file."""
        logger.info(f"Reading PDF for file: {filename} ...")

        with tempfile.TemporaryDirectory() as temp_dir:
            _image_paths = convert_pdf_to_image(
                filename,
                pdf_image_dpi,
                output_folder=temp_dir,
                path_only=True,
            )
            image_paths = cast(List[str], _image_paths)
            number_of_pages = len(image_paths)
            pages: List[PageLayout] = []
            if fixed_layouts is None:
                fixed_layouts = [None for _ in range(0, number_of_pages)]
            for i, (image_path, fixed_layout) in enumerate(zip(image_paths, fixed_layouts)):
                # NOTE(robinson) - In the future, maybe we detect the page number and default
                # to the index if it is not detected
                with Image.open(image_path) as image:
                    page = PageLayout.from_image(
                        image,
                        number=i + 1,
                        document_filename=filename,
                        fixed_layout=fixed_layout,
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
        fixed_layout: Optional[List[TextRegion]] = None,
        **kwargs,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from an image file."""
        logger.info(f"Reading image file: {filename} ...")
        try:
            image = Image.open(filename)
            format = image.format
            images: list[Image.Image] = []
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
        for i, image in enumerate(images):  # type: ignore
            page = PageLayout.from_image(
                image,
                image_path=filename,
                number=i,
                detection_model=detection_model,
                element_extraction_model=element_extraction_model,
                fixed_layout=fixed_layout,
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
        image_metadata: Optional[dict] = None,
        image_path: Optional[Union[str, PurePath]] = None,  # TODO: Deprecate
        document_filename: Optional[Union[str, PurePath]] = None,
        detection_model: Optional[UnstructuredObjectDetectionModel] = None,
        element_extraction_model: Optional[UnstructuredElementExtractionModel] = None,
    ):
        if detection_model is not None and element_extraction_model is not None:
            raise ValueError("Only one of detection_model and extraction_model should be passed.")
        self.image: Optional[Image.Image] = image
        if image_metadata is None:
            image_metadata = {}
        self.image_metadata = image_metadata
        self.image_path = image_path
        self.image_array: Union[np.ndarray[Any, Any], None] = None
        self.document_filename = document_filename
        self.number = number
        self.detection_model = detection_model
        self.element_extraction_model = element_extraction_model
        self.elements: Collection[LayoutElement] = []
        self.elements_array: LayoutElements | None = None
        # NOTE(alan): Dropped LocationlessLayoutElement that was created for chipper - chipper has
        # locations now and if we need to support LayoutElements without bounding boxes we can make
        # the bbox property optional

    def __str__(self) -> str:
        return "\n\n".join([str(element) for element in self.elements])

    def get_elements_using_image_extraction(
        self,
        inplace=True,
    ) -> Optional[List[LayoutElement]]:
        """Uses end-to-end text element extraction model to extract the elements on the page."""
        if self.element_extraction_model is None:
            raise ValueError(
                "Cannot get elements using image extraction, no image extraction model defined",
            )
        assert self.image is not None
        elements = self.element_extraction_model(self.image)
        if inplace:
            self.elements = elements
            return None
        return elements

    def get_elements_with_detection_model(
        self,
        inplace: bool = True,
        array_only: bool = False,
    ) -> Optional[List[LayoutElement]]:
        """Uses specified model to detect the elements on the page."""
        if self.detection_model is None:
            model = get_model()
            if isinstance(model, UnstructuredObjectDetectionModel):
                self.detection_model = model
            else:
                raise NotImplementedError("Default model should be a detection model")

        # NOTE(mrobinson) - We'll want make this model inference step some kind of
        # remote call in the future.
        assert self.image is not None
        inferred_layout: LayoutElements = self.detection_model(self.image)
        inferred_layout = self.detection_model.deduplicate_detected_elements(
            inferred_layout,
        )

        if inplace:
            self.elements_array = inferred_layout
            if not array_only:
                self.elements = inferred_layout.as_list()
            return None

        return inferred_layout.as_list()

    def _get_image_array(self) -> Union[np.ndarray[Any, Any], None]:
        """Converts the raw image into a numpy array."""
        if self.image_array is None:
            if self.image:
                self.image_array = np.array(self.image)
            else:
                image = Image.open(self.image_path)  # type: ignore
                self.image_array = np.array(image)
        return self.image_array

    def annotate(
        self,
        colors: Optional[Union[List[str], str]] = None,
        image_dpi: int = 200,
        annotation_data: Optional[dict[str, dict]] = None,
        add_details: bool = False,
        sources: Optional[List[str]] = None,
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
                if sources is None or el.source in sources:
                    img = draw_bbox(img, el, color=color, details=add_details)
        else:
            for attribute, style in annotation_data.items():
                if hasattr(self, attribute) and getattr(self, attribute):
                    color = style["color"]
                    width = style["width"]
                    for region in getattr(self, attribute):
                        required_source = getattr(region, "source", None)
                        if (sources is None) or (required_source in sources):
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
        fixed_layout: Optional[List[TextRegion]] = None,
    ):
        """Creates a PageLayout from an already-loaded PIL Image."""

        page = cls(
            number=number,
            image=image,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
        )
        # FIXME (yao): refactor the other methods so they all return elements like the third route
        if page.element_extraction_model is not None:
            page.get_elements_using_image_extraction()
        elif fixed_layout is None:
            page.get_elements_with_detection_model()
        else:
            page.elements = []

        page.image_metadata = {
            "format": page.image.format if page.image else None,
            "width": page.image.width if page.image else None,
            "height": page.image.height if page.image else None,
        }
        page.image_path = os.path.abspath(image_path) if image_path else None
        page.document_filename = os.path.abspath(document_filename) if document_filename else None

        # Clear the image to save memory
        page.image = None

        return page


def process_data_with_model(
    data: BinaryIO,
    model_name: Optional[str],
    **kwargs: Any,
) -> DocumentLayout:
    """Process PDF as file-like object `data` into a `DocumentLayout`.

    Uses the model identified by `model_name`.
    """
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        file_path = os.path.join(tmp_dir_path, "document.pdf")
        with open(file_path, "wb") as f:
            f.write(data.read())
            f.flush()
        layout = process_file_with_model(
            file_path,
            model_name,
            **kwargs,
        )

    return layout


def process_file_with_model(
    filename: str,
    model_name: Optional[str],
    is_image: bool = False,
    fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
    pdf_image_dpi: int = 200,
    **kwargs: Any,
) -> DocumentLayout:
    """Processes pdf file with name filename into a DocumentLayout by using a model identified by
    model_name."""

    model = get_model(model_name, **kwargs)
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
            **kwargs,
        )
        if is_image
        else DocumentLayout.from_file(
            filename,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
            fixed_layouts=fixed_layouts,
            pdf_image_dpi=pdf_image_dpi,
            **kwargs,
        )
    )
    return layout


def convert_pdf_to_image(
    filename: str,
    dpi: int = 200,
    output_folder: Optional[Union[str, PurePath]] = None,
    path_only: bool = False,
) -> Union[List[Image.Image], List[str]]:
    """Get the image renderings of the pdf pages using pdf2image"""

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

    return images
