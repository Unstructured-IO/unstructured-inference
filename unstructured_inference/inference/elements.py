from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from layoutparser.elements.layout import TextBlock


@dataclass
class Rectangle:
    x1: int
    y1: int
    x2: int
    y2: int

    def pad(self, padding: int):
        """Increases (or decreases, if padding is negative) the size of the rectangle by extending
        the boundary outward (resp. inward)."""
        out_object = deepcopy(self)
        out_object.x1 -= padding
        out_object.y1 -= padding
        out_object.x2 += padding
        out_object.y2 += padding
        return out_object

    @property
    def width(self):
        """Width of rectangle"""
        return self.x2 - self.x1

    @property
    def height(self):
        """Height of rectangle"""
        return self.y2 - self.y1

    def is_disjoint(self, other: Rectangle):
        """Checks whether this rectangle is disjoint from another rectangle."""
        return ((self.x2 < other.x1) or (self.x1 > other.x2)) and (
            (self.y2 < other.y1) or (self.y1 > other.y2)
        )

    def intersects(self, other: Rectangle):
        """Checks whether this rectangle intersects another rectangle."""
        return not self.is_disjoint(other)

    def is_in(self, other: Rectangle, error_margin: Optional[int] = None):
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


@dataclass
class TextRegion(Rectangle):
    text: Optional[str] = None

    def __str__(self) -> str:
        return str(self.text)


class ImageTextRegion(TextRegion):
    pass


@dataclass
class LayoutElement(TextRegion):
    type: Optional[str] = None

    def to_dict(self) -> dict:
        """Converts the class instance to dictionary form."""
        return self.__dict__

    @classmethod
    def from_region(cls, region: Rectangle):
        """Create LayoutElement from superclass."""
        x1, y1, x2, y2 = region.x1, region.y1, region.x2, region.y2
        text = region.text if hasattr(region, "text") else None
        type = region.type if hasattr(region, "type") else None
        return cls(x1, y1, x2, y2, text, type)

    @classmethod
    def from_lp_textblock(cls, textblock: TextBlock):
        """Create LayoutElement from layoutparser TextBlock object."""
        x1, y1, x2, y2 = textblock.coordinates
        text = textblock.text
        type = textblock.type
        return cls(x1, y1, x2, y2, text, type)
