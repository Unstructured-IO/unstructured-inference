from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path, PurePath
from threading import Lock
from typing import BinaryIO, Optional, Union

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from unstructured_inference.config import inference_config
from unstructured_inference.constants import PDF_POINTS_PER_INCH

_pdfium_lock = Lock()


class PdfRenderTooLargeError(ValueError):
    pass


def _check_pdf_render_max_pixels(page, page_number: int, scale: float, maximum: int) -> None:
    if maximum <= 0:
        return

    rendered_width = math.ceil(page.get_width() * scale)
    rendered_height = math.ceil(page.get_height() * scale)
    rendered_pixels = rendered_width * rendered_height

    if rendered_pixels > maximum:
        raise PdfRenderTooLargeError(
            "PDF page would render to too many pixels for safe processing: "
            f"page={page_number}, pixels={rendered_pixels}, maximum={maximum}. "
            "Try splitting the PDF, reducing the page dimensions, or using a lower render DPI.",
        )


@lru_cache(maxsize=1)
def _get_pdfium_module():
    import pypdfium2 as pdfium

    return pdfium


def convert_pdf_to_image(
    filename: Optional[str] = None,
    file: Optional[Union[bytes, BinaryIO]] = None,
    dpi: int = 200,
    output_folder: Optional[Union[str, PurePath]] = None,
    path_only: bool = False,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    password: Optional[str] = None,
    pdf_render_max_pixels_per_page: Optional[int] = None,
) -> Union[list[Image.Image], list[str]]:
    """Render PDF pages to PIL images or saved PNGs using pypdfium2.

    This is the single source of truth for PDF→image rendering across unstructured
    and unstructured-inference. Callers should pass their own DPI value explicitly.
    """
    if path_only and not output_folder:
        raise ValueError("output_folder must be specified if path_only is true")
    if filename is None and file is None:
        raise ValueError("Either filename or file must be provided")
    if output_folder:
        assert Path(output_folder).exists()
        assert Path(output_folder).is_dir()

    scale = dpi / PDF_POINTS_PER_INCH
    if pdf_render_max_pixels_per_page is None:
        pdf_render_max_pixels_per_page = inference_config.PDF_RENDER_MAX_PIXELS_PER_PAGE
    pdfium = _get_pdfium_module()

    with _pdfium_lock:
        pdf = pdfium.PdfDocument(filename or file, password=password)
        n_pages = len(pdf)

    try:
        images: dict[int, Image.Image] = {}
        filenames: list[str] = []
        for i in range(n_pages):
            page_num = i + 1
            if first_page is not None and page_num < first_page:
                continue
            if last_page is not None and page_num > last_page:
                break

            with _pdfium_lock:
                page = pdf[i]
                try:
                    _check_pdf_render_max_pixels(
                        page=page,
                        page_number=page_num,
                        scale=scale,
                        maximum=pdf_render_max_pixels_per_page,
                    )
                    bitmap = page.render(
                        scale=scale,
                        no_smoothtext=False,
                        no_smoothimage=False,
                        no_smoothpath=False,
                        optimize_mode="print",
                    )
                    try:
                        pil_image = bitmap.to_pil()
                    finally:
                        bitmap.close()

                    rotation = page.get_rotation()
                    if rotation:
                        pil_image = pil_image.rotate(rotation, expand=True)
                    pil_image.info["pdf_rotation"] = rotation

                finally:
                    page.close()

            if output_folder:
                fn: str = os.path.join(str(output_folder), f"page_{page_num}.png")

                png_meta = PngInfo()
                png_meta.add_text("pdf_rotation", str(rotation))
                pil_image.save(
                    fn,
                    format="PNG",
                    compress_level=1,
                    optimize=False,
                    pnginfo=png_meta,
                )
                filenames.append(fn)
                if not path_only:
                    images[page_num] = pil_image
            else:
                images[page_num] = pil_image
    finally:
        with _pdfium_lock:
            pdf.close()

    if path_only:
        return filenames
    return list(images.values())
