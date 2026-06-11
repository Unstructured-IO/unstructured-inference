from __future__ import annotations

import numpy as np
import pypdfium2 as pdfium
from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

from unstructured_inference.inference import pdf_image

# Page geometry and the single form field used by the synthetic fixture below.
PAGE_WIDTH, PAGE_HEIGHT = 612, 792
# Widget rectangle in PDF user space (origin bottom-left): x1, y1, x2, y2.
FIELD_RECT = (40, 700, 320, 724)
RENDER_DPI = 200


def _build_acroform_pdf(path: str) -> None:
    """Write a 1-page PDF whose only mark is a filled text form field.

    The page content stream is empty; the field's value is drawn solely by the widget
    annotation's appearance stream (``/AP /N``). pdfium only paints widget appearances
    after the form environment is initialized, so this fixture renders blank unless
    ``convert_pdf_to_image`` calls ``init_forms()``.
    """
    writer = PdfWriter()
    writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page = writer.pages[0]

    # Helvetica, referenced by both the appearance stream and the AcroForm default resources.
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    font_ref = writer._add_object(font)
    fonts = DictionaryObject()
    fonts[NameObject("/Helv")] = font_ref
    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts

    # Appearance stream that draws the field value inside the widget box.
    rect_w, rect_h = FIELD_RECT[2] - FIELD_RECT[0], FIELD_RECT[3] - FIELD_RECT[1]
    appearance = DecodedStreamObject()
    appearance.set_data(b"/Tx BMC BT /Helv 14 Tf 0 g 2 6 Td (FORMVALUE777) Tj ET EMC")
    appearance[NameObject("/Type")] = NameObject("/XObject")
    appearance[NameObject("/Subtype")] = NameObject("/Form")
    appearance[NameObject("/BBox")] = ArrayObject(
        [NumberObject(0), NumberObject(0), NumberObject(rect_w), NumberObject(rect_h)]
    )
    appearance[NameObject("/Resources")] = resources
    appearance_ref = writer._add_object(appearance)
    appearance_dict = DictionaryObject()
    appearance_dict[NameObject("/N")] = appearance_ref

    widget = DictionaryObject()
    widget[NameObject("/Type")] = NameObject("/Annot")
    widget[NameObject("/Subtype")] = NameObject("/Widget")
    widget[NameObject("/FT")] = NameObject("/Tx")
    widget[NameObject("/T")] = TextStringObject("name")
    widget[NameObject("/V")] = TextStringObject("FORMVALUE777")
    widget[NameObject("/Rect")] = ArrayObject([NumberObject(c) for c in FIELD_RECT])
    widget[NameObject("/AP")] = appearance_dict
    widget_ref = writer._add_object(widget)
    page[NameObject("/Annots")] = ArrayObject([widget_ref])

    acro_form = DictionaryObject()
    acro_form[NameObject("/Fields")] = ArrayObject([widget_ref])
    default_resources = DictionaryObject()
    default_resources[NameObject("/Font")] = fonts
    acro_form[NameObject("/DR")] = default_resources
    writer._root_object[NameObject("/AcroForm")] = writer._add_object(acro_form)

    with open(path, "wb") as f:
        writer.write(f)


def _field_region_dark_pixels(img) -> int:
    """Count dark pixels inside the form field's rectangle in the rendered image."""
    gray = img.convert("L")
    scale_x = gray.width / PAGE_WIDTH
    scale_y = gray.height / PAGE_HEIGHT
    x0, y0, x1, y1 = FIELD_RECT
    # PDF user space is bottom-up; image space is top-down.
    box = (
        int(x0 * scale_x),
        int((PAGE_HEIGHT - y1) * scale_y),
        int(x1 * scale_x),
        int((PAGE_HEIGHT - y0) * scale_y),
    )
    crop = np.array(gray.crop(box))
    return int(np.count_nonzero(crop < 128))


def test_convert_pdf_to_image_renders_acroform_field_value(tmp_path):
    """Filled form-field values are painted into the rendered page image."""
    pdf_path = str(tmp_path / "form.pdf")
    _build_acroform_pdf(pdf_path)

    img = pdf_image.convert_pdf_to_image(filename=pdf_path, dpi=RENDER_DPI)[0]

    assert _field_region_dark_pixels(img) > 100, "Expected the form field value to be rendered"


def test_convert_pdf_to_image_drops_form_field_without_init_forms(tmp_path, monkeypatch):
    """Control: without init_forms() the widget appearance is not painted.

    Patching init_forms() to a no-op reproduces the pre-fix behavior and proves the
    rendered field value in the test above comes specifically from initializing the
    form-fill environment, not from the page content stream (which is empty here).
    """
    pdf_path = str(tmp_path / "form.pdf")
    _build_acroform_pdf(pdf_path)

    monkeypatch.setattr(pdfium.PdfDocument, "init_forms", lambda self, *a, **k: None)
    img = pdf_image.convert_pdf_to_image(filename=pdf_path, dpi=RENDER_DPI)[0]

    assert _field_region_dark_pixels(img) == 0, "Field should be blank without form init"
