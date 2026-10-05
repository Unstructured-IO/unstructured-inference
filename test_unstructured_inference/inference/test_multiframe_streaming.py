import numpy as np
import pytest
from PIL import Image

from unstructured_inference.inference import layout


def test_tiff_frames_are_processed_before_decoding_the_next(tmp_path, monkeypatch):
    path = tmp_path / "frames.tiff"
    frames = [Image.new("RGB", (20, 10), color) for color in ("red", "green", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:], compression="tiff_lzw")
    converted = []
    processed = []
    original_convert = Image.Image.convert
    original_from_image = layout.PageLayout.from_image.__func__

    def convert(self, *args, **kwargs):
        result = original_convert(self, *args, **kwargs)
        converted.append(result)
        return result

    def from_image(cls, image, **kwargs):
        assert len(converted) == len(processed) + 1
        processed.append(np.array(image))
        return original_from_image(cls, image, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", convert)
    monkeypatch.setattr(layout.PageLayout, "from_image", classmethod(from_image))
    document = layout.DocumentLayout.from_image_file(str(path), fixed_layout=[])
    assert [page.number for page in document.pages] == [0, 1, 2]
    for page, expected, actual in zip(document.pages, frames, processed):
        np.testing.assert_array_equal(actual, np.array(expected))
        assert page.image_metadata["format"] == "TIFF"
        assert (page.image_metadata["width"], page.image_metadata["height"]) == expected.size
        assert page.image is None
    for image in converted:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))


def test_source_and_frame_close_when_inference_fails(tmp_path, monkeypatch):
    path = tmp_path / "image.tiff"
    Image.new("RGB", (10, 10)).save(path)
    opened = []
    converted = []
    original_open = Image.open

    def open_image(*args, **kwargs):
        image = original_open(*args, **kwargs)
        opened.append(image)
        return image

    def fail(cls, image, **kwargs):
        converted.append(image)
        raise RuntimeError("inference failed")

    monkeypatch.setattr(Image, "open", open_image)
    monkeypatch.setattr(layout.PageLayout, "from_image", classmethod(fail))
    with pytest.raises(RuntimeError, match="inference failed"):
        layout.DocumentLayout.from_image_file(str(path))
    assert opened[0].fp is None
    with pytest.raises(ValueError, match="closed image"):
        converted[0].getpixel((0, 0))
