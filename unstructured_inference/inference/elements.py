from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Optional, Union, Sequence, List
import unicodedata

import numpy as np
from PIL import Image
from scipy.sparse.csgraph import connected_components

from unstructured_inference.logger import logger
from unstructured_inference.models import tesseract

# When extending the boundaries of a PDF object for the purpose of determining which other elements
# should be considered in the same text region, we use a relative distance based on some fraction of
# the block height (typically character height). This is the fraction used for the horizontal
# extension applied to the left and right sides.
H_PADDING_COEF = 0.4
# Same as above but the vertical extension.
V_PADDING_COEF = 0.3


@dataclass
class Rectangle:
    x1: Union[int, float]
    y1: Union[int, float]
    x2: Union[int, float]
    y2: Union[int, float]

    def pad(self, padding: Union[int, float]):
        """Increases (or decreases, if padding is negative) the size of the rectangle by extending
        the boundary outward (resp. inward)."""
        out_object = self.hpad(padding).vpad(padding)
        return out_object

    def hpad(self, padding: Union[int, float]):
        """Increases (or decreases, if padding is negative) the size of the rectangle by extending
        the left and right sides of the boundary outward (resp. inward)."""
        out_object = deepcopy(self)
        out_object.x1 -= padding
        out_object.x2 += padding
        return out_object

    def vpad(self, padding: Union[int, float]):
        """Increases (or decreases, if padding is negative) the size of the rectangle by extending
        the top and bottom of the boundary outward (resp. inward)."""
        out_object = deepcopy(self)
        out_object.y1 -= padding
        out_object.y2 += padding
        return out_object

    @property
    def width(self) -> Union[int, float]:
        """Width of rectangle"""
        return self.x2 - self.x1

    @property
    def height(self) -> Union[int, float]:
        """Height of rectangle"""
        return self.y2 - self.y1

    def is_disjoint(self, other: Rectangle) -> bool:
        """Checks whether this rectangle is disjoint from another rectangle."""
        return not self.intersects(other)

    def intersects(self, other: Rectangle) -> bool:
        """Checks whether this rectangle intersects another rectangle."""
        return intersections(self, other)[0, 1]

    def is_in(self, other: Rectangle, error_margin: Optional[Union[int, float]] = None) -> bool:
        """Checks whether this rectangle is contained within another rectangle."""
        if error_margin is not None:
            padded_other = other.pad(error_margin)
        else:
            padded_other = other
        return all(
            [
                (self.x1 >= padded_other.x1),
                (self.x2 <= padded_other.x2),
                (self.y1 >= padded_other.y1),
                (self.y2 <= padded_other.y2),
            ]
        )

    @property
    def coordinates(self):
        """Gets coordinates of the rectangle"""
        return ((self.x1, self.y1), (self.x1, self.y2), (self.x2, self.y2), (self.x2, self.y1))


def minimal_containing_region(*regions: Rectangle) -> Rectangle:
    """Returns the smallest rectangular region that contains all regions passed"""
    x1 = min(region.x1 for region in regions)
    y1 = min(region.y1 for region in regions)
    x2 = max(region.x2 for region in regions)
    y2 = max(region.y2 for region in regions)

    # Return most specialized class of which that every region is a subclass
    def least_common_superclass(*instances):
        mros = (type(ins).mro() for ins in instances)
        mro = next(mros)
        common = set(mro).intersection(*mros)
        return next((x for x in mro if x in common), Rectangle)

    cls = least_common_superclass(*regions)
    return cls(x1, y1, x2, y2)


def partition_groups_from_regions(regions: Sequence[Rectangle]) -> List[List[Rectangle]]:
    """Partitions regions into groups of regions based on proximity. Returns list of lists of
    regions, each list corresponding with a group"""
    padded_regions = [
        r.vpad(r.height * V_PADDING_COEF).hpad(r.height * H_PADDING_COEF) for r in regions
    ]

    intersection_mtx = intersections(*padded_regions)

    _, group_nums = connected_components(intersection_mtx)
    groups: List[List[Rectangle]] = [[] for _ in range(max(group_nums) + 1)]
    for region, group_num in zip(regions, group_nums):
        groups[group_num].append(region)

    return groups


def intersections(*rects: Rectangle):
    """Returns a square boolean matrix of intersections of an arbitrary number of rectangles, i.e.
    the ijth entry of the matrix is True if and only if the ith Rectangle and jth Rectangle
    intersect."""
    coords = np.stack([[[r.x1, r.y1], [r.x2, r.y2]] for r in rects], axis=-1)

    (x1s, y1s), (x2s, y2s) = coords

    # Use broadcasting to get comparison matrices.
    # For Rectangles r1 and r2, any of the following conditions makes the rectangles disjoint:
    # r1.x1 > r2.x2
    # r1.y1 > r2.y2
    # r2.x1 > r1.x2
    # r2.y1 > r1.y2
    # Then we take the complement (~) of the disjointness matrix to get the intersection matrix.
    intersections = ~(
        (x1s[None] > x2s[..., None])
        | (y1s[None] > y2s[..., None])
        | (x1s[None] > x2s[..., None]).T
        | (y1s[None] > y2s[..., None]).T
    )

    return intersections


@dataclass
class TextRegion(Rectangle):
    text: Optional[str] = None

    def __str__(self) -> str:
        return str(self.text)

    def extract_text(
        self,
        objects: Optional[List[TextRegion]],
        image: Optional[Image.Image] = None,
        extract_tables: bool = False,
        ocr_strategy: str = "auto",
    ) -> str:
        """Extracts text contained in region."""
        if self.text is not None:
            # If block text is already populated, we'll assume it's correct
            text = self.text
        elif objects is not None:
            text = aggregate_by_block(self, image, objects, ocr_strategy)
        elif image is not None:
            if ocr_strategy != "never":
                # We don't have anything to go on but the image itself, so we use OCR
                text = ocr(self, image)
            else:
                text = ""
        else:
            raise ValueError(
                "Got arguments image and layout as None, at least one must be populated to use for "
                "text extraction."
            )
        return text


class EmbeddedTextRegion(TextRegion):
    def extract_text(
        self,
        objects: Optional[List[TextRegion]],
        image: Optional[Image.Image] = None,
        extract_tables: bool = False,
        ocr_strategy: str = "auto",
    ) -> str:
        """Extracts text contained in region."""
        if self.text is None:
            return ""
        else:
            return self.text


class ImageTextRegion(TextRegion):
    def extract_text(
        self,
        objects: Optional[List[TextRegion]],
        image: Optional[Image.Image] = None,
        extract_tables: bool = False,
        ocr_strategy: str = "auto",
    ) -> str:
        """Extracts text contained in region."""
        if self.text is None:
            if ocr_strategy == "never" or image is None:
                return ""
            else:
                return ocr(self, image)
        else:
            return super().extract_text(objects, image, extract_tables, ocr_strategy)


def ocr(text_block: TextRegion, image: Image.Image) -> str:
    """Runs a cropped text block image through and OCR agent."""
    logger.debug("Running OCR on text block ...")
    tesseract.load_agent()
    padded_block = text_block.pad(12)
    cropped_image = image.crop((padded_block.x1, padded_block.y1, padded_block.x2, padded_block.y2))
    return tesseract.ocr_agent.detect(cropped_image)


def needs_ocr(
    region: TextRegion,
    pdf_objects: List[TextRegion],
    ocr_strategy: str,
) -> bool:
    """Logic to determine whether ocr is needed to extract text from given region."""
    if ocr_strategy == "force":
        return True
    elif ocr_strategy == "auto":
        image_objects = [obj for obj in pdf_objects if isinstance(obj, ImageTextRegion)]
        word_objects = [obj for obj in pdf_objects if isinstance(obj, EmbeddedTextRegion)]
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
            return True
        else:
            return False
    else:
        return False


def aggregate_by_block(
    text_region: TextRegion,
    image: Optional[Image.Image],
    pdf_objects: List[TextRegion],
    ocr_strategy: str = "auto",
) -> str:
    """Extracts the text aggregated from the elements of the given layout that lie within the given
    block."""
    if image is not None and needs_ocr(text_region, pdf_objects, ocr_strategy):
        text = ocr(text_region, image)
    else:
        filtered_blocks = [obj for obj in pdf_objects if obj.is_in(text_region, error_margin=5)]
        for little_block in filtered_blocks:
            if image is not None and needs_ocr(little_block, pdf_objects, ocr_strategy):
                little_block.text = ocr(little_block, image)
        text = " ".join([x.text for x in filtered_blocks if x.text])
    text = remove_control_characters(text)
    return text


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


def remove_control_characters(text: str) -> str:
    """Removes control characters from text."""
    out_text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
    return out_text
