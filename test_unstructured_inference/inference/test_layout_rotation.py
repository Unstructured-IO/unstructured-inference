from __future__ import annotations

from typing import Optional, Union

from PIL import Image

from unstructured_inference.inference import pdf_image


class _FakeBitmap:
    def __init__(self, image: Image.Image):
        self._image = image

    def to_pil(self) -> Image.Image:
        return self._image.copy()

    def close(self) -> None:
        return None


class _FakePage:
    def __init__(self, image: Image.Image, rotation: int):
        self._image = image
        self._rotation = rotation

    def render(self, **kwargs) -> _FakeBitmap:
        return _FakeBitmap(self._image)

    def get_rotation(self) -> int:
        return self._rotation

    def close(self) -> None:
        return None


class _FakePdfDocument:
    def __init__(self, source: Optional[Union[str, bytes]], password: Optional[str] = None):
        del source, password
        self._pages = [_FakePage(Image.new("RGB", (200, 100), color="white"), rotation=90)]

    def __len__(self) -> int:
        return len(self._pages)

    def __getitem__(self, index: int) -> _FakePage:
        return self._pages[index]

    def close(self) -> None:
        return None


def test_convert_pdf_to_image_applies_rotation(monkeypatch):
    """Pages with /Rotate metadata are rendered upright."""
    fake_pdfium = type("_FakePdfiumModule", (), {"PdfDocument": _FakePdfDocument})
    monkeypatch.setattr(pdf_image, "_get_pdfium_module", lambda: fake_pdfium)

    result = pdf_image.convert_pdf_to_image(file=b"%PDF-1.7\n", dpi=72)

    assert len(result) == 1
    image = result[0]
    assert image.size == (100, 200), f"Expected portrait after rotation, got {image.size}"
