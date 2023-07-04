from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, List, Optional

from layoutparser.elements.layout import TextBlock
from PIL import Image

from unstructured_inference.inference.elements import (
    ImageTextRegion,
    Rectangle,
    TextRegion,
    grow_region_to_match_region,
    region_bounding_boxes_are_almost_the_same,
)
from unstructured_inference.models import tables


@dataclass
class LayoutElement(TextRegion):
    type: Optional[str] = None

    def extract_text(
        self,
        objects: Optional[Collection[TextRegion]],
        image: Optional[Image.Image] = None,
        extract_tables: bool = False,
        ocr_strategy: str = "auto",
        ocr_languages: str = "eng",
    ):
        """Extracts text contained in region"""
        text = super().extract_text(
            objects=objects,
            image=image,
            extract_tables=extract_tables,
            ocr_strategy=ocr_strategy,
            ocr_languages=ocr_languages,
        )
        if extract_tables and self.type == "Table":
            self.text_as_html = interpret_table_block(self, image)
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


def interpret_table_block(text_block: TextRegion, image: Image.Image) -> str:
    """Extract the contents of a table."""
    tables.load_agent()
    if tables.tables_agent is None:
        raise RuntimeError("Unable to load table extraction agent.")
    padded_block = text_block.pad(12)
    cropped_image = image.crop((padded_block.x1, padded_block.y1, padded_block.x2, padded_block.y2))
    return tables.tables_agent.predict(cropped_image)


def merge_inferred_layout_with_extracted_layout(
    inferred_layout: Collection[LayoutElement],
    extracted_layout: Collection[TextRegion],
    same_region_threshold: float = 0.75,
    subregion_threshold: float = 0.75,
) -> List[LayoutElement]:
    """Merge two layouts to produce a single layout."""
    extracted_elements_to_add: List[TextRegion] = []
    inferred_regions_to_remove = []
    for extracted_region in extracted_layout:
        region_matched = False
        for inferred_region in inferred_layout:
            if inferred_region.intersects(extracted_region):
                same_bbox = region_bounding_boxes_are_almost_the_same(
                    inferred_region,
                    extracted_region,
                    same_region_threshold,
                )
                inferred_is_subregion_of_extracted = inferred_region.is_almost_subregion_of(
                    extracted_region,
                    subregion_threshold=subregion_threshold,
                )
                extracted_is_image = isinstance(extracted_region, ImageTextRegion)
                inferred_is_text = inferred_region.type not in (
                    "Figure",
                    "Image",
                    "PageBreak",
                    "Table",
                )
                extracted_is_subregion_of_inferred = extracted_region.is_almost_subregion_of(
                    inferred_region,
                    subregion_threshold=subregion_threshold,
                )
                either_region_is_subregion_of_other = (
                    inferred_is_subregion_of_extracted or extracted_is_subregion_of_inferred
                )
                if same_bbox:
                    # Looks like these represent the same region
                    grow_region_to_match_region(inferred_region, extracted_region)
                    inferred_region.text = extracted_region.text
                    region_matched = True
                elif extracted_is_subregion_of_inferred and inferred_is_text and extracted_is_image:
                    grow_region_to_match_region(inferred_region, extracted_region)
                    region_matched = True
                elif either_region_is_subregion_of_other and inferred_region.type != "Table":
                    inferred_regions_to_remove.append(inferred_region)
        if not region_matched:
            extracted_elements_to_add.append(extracted_region)
    # Need to classify the extracted layout elements we're keeping.
    categorized_extracted_elements_to_add = [
        LayoutElement(
            el.x1,
            el.y1,
            el.x2,
            el.y2,
            text=el.text,
            type="Image" if isinstance(el, ImageTextRegion) else "UncategorizedText",
        )
        for el in extracted_elements_to_add
    ]
    out_layout = categorized_extracted_elements_to_add + [
        region for region in inferred_layout if region not in inferred_regions_to_remove
    ]
    return out_layout


# NOTE(alan): The right way to do this is probably to rewrite LayoutElement as well as the different
# Region types to not subclass Rectangle, and instead have an optional bbox property that is a
# Rectangle. I or someone else will have to get to that later.
@dataclass
class LocationlessLayoutElement:
    text: Optional[str]
    type: Optional[str]

    def to_dict(self) -> dict:
        """Converts the class instance to dictionary form."""
        out_dict = {
            "text": self.text,
            "type": self.type,
        }
        return out_dict
