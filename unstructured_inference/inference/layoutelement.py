from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from layoutparser.elements.layout import TextBlock
from PIL import Image

from unstructured_inference.inference.elements import Rectangle, TextRegion
from unstructured_inference.models import tables


@dataclass
class LayoutElement(TextRegion):
    type: Optional[str] = None

    def extract_text(
        self,
        objects: Optional[List[TextRegion]],
        image: Optional[Image.Image] = None,
        extract_tables: bool = False,
        ocr_strategy: str = "auto",
    ):
        """Extracts text contained in region"""
        if self.text is not None:
            # If block text is already populated, we'll assume it's correct
            text = self.text
        elif extract_tables and isinstance(self, LayoutElement) and self.type == "Table":
            text = interprete_table_block(self, image)
        else:
            text = super().extract_text(
                objects=objects,
                image=image,
                extract_tables=extract_tables,
                ocr_strategy=ocr_strategy,
            )
        return text

    def to_dict(self) -> dict:
        """Converts the class instance to dictionary form."""
        out_dict = {
            "coordinates": self.coordinates,
            "text": self.text,
            "type": self.type,
        }
        return out_dict

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


def interprete_table_block(text_block: TextRegion, image: Image.Image) -> str:
    """Extract the contents of a table."""
    tables.load_agent()
    if tables.tables_agent is None:
        raise RuntimeError("Unable to load table extraction agent.")
    padded_block = text_block.pad(12)
    cropped_image = image.crop((padded_block.x1, padded_block.y1, padded_block.x2, padded_block.y2))
    return tables.tables_agent.predict(cropped_image)
