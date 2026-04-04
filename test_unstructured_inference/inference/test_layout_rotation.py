from __future__ import annotations

import numpy as np

from unstructured_inference.inference import pdf_image


def test_convert_pdf_to_image_applies_rotation():
    """Pages with /Rotate metadata are rendered upright."""
    result = pdf_image.convert_pdf_to_image(filename="sample-docs/rotated-page-90.pdf", dpi=72)
    assert len(result) == 1
    img = result[0]
    # The PDF has /Rotate=90 on a landscape page (width > height in PDF units).
    # Without rotation fix the rendered image would be landscape; with the fix it's portrait.
    assert img.height > img.width, f"Expected portrait after rotation, got {img.size}"

    # Fixture contract: rotated-page-90.pdf has visible dark text in the upper half when upright.
    # Use relative dark-pixel counts to reduce sensitivity to minor renderer differences.
    gray = np.array(img.convert("L"))
    split = gray.shape[0] // 2
    top_dark_pixels = int(np.count_nonzero(gray[:split] < 245))
    bottom_dark_pixels = int(np.count_nonzero(gray[split:] < 245))

    assert top_dark_pixels > 0, "Expected text pixels in upper half of upright page"
    assert top_dark_pixels > max(bottom_dark_pixels * 10, 50), (
        "Expected substantially more dark pixels in upper half for upright orientation; "
        f"got top={top_dark_pixels}, bottom={bottom_dark_pixels}"
    )
