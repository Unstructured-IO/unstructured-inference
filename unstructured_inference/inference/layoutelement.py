from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, List, Optional, Union

import numpy as np
from layoutparser.elements.layout import TextBlock
from pandas import DataFrame
from PIL import Image

from unstructured_inference.config import inference_config
from unstructured_inference.constants import (
    FULL_PAGE_REGION_THRESHOLD,
    Source,
)
from unstructured_inference.inference.elements import (
    ImageTextRegion,
    Rectangle,
    TextRegion,
    grow_region_to_match_region,
    region_bounding_boxes_are_almost_the_same,
)


@dataclass
class LayoutElement(TextRegion):
    type: Optional[str] = None
    prob: Optional[float] = None
    image_path: Optional[str] = None

    def extract_text(
        self,
        objects: Optional[Collection[TextRegion]],
        image: Optional[Image.Image] = None,
        extract_tables: bool = False,
    ):
        """Extracts text contained in region"""
        text = super().extract_text(
            objects=objects,
            extract_tables=extract_tables,
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
            "prob": self.prob,
            "source": self.source,
        }
        return out_dict

    @classmethod
    def from_region(cls, region: Rectangle):
        """Create LayoutElement from superclass."""
        x1, y1, x2, y2 = region.x1, region.y1, region.x2, region.y2
        text = region.text if hasattr(region, "text") else None
        type = region.type if hasattr(region, "type") else None
        prob = region.prob if hasattr(region, "prob") else None
        source = region.source if hasattr(region, "source") else None
        return cls(x1, y1, x2, y2, text=text, source=source, type=type, prob=prob)

    @classmethod
    def from_lp_textblock(cls, textblock: TextBlock):
        """Create LayoutElement from layoutparser TextBlock object."""
        x1, y1, x2, y2 = textblock.coordinates
        text = textblock.text
        type = textblock.type
        prob = textblock.score
        return cls(x1, y1, x2, y2, text=text, source=Source.DETECTRON2_LP, type=type, prob=prob)


def interpret_table_block(text_block: TextRegion, image: Image.Image) -> str:
    """Extract the contents of a table."""
    from unstructured_inference.models import tables

    tables.load_agent()
    if tables.tables_agent is None:
        raise RuntimeError("Unable to load table extraction agent.")
    padded_block = text_block.pad(inference_config.TABLE_IMAGE_CROP_PAD)
    cropped_image = image.crop((padded_block.x1, padded_block.y1, padded_block.x2, padded_block.y2))
    return tables.tables_agent.predict(cropped_image)


def merge_inferred_layout_with_extracted_layout(
    inferred_layout: Collection[LayoutElement],
    extracted_layout: Collection[TextRegion],
    page_image_size: tuple,
    same_region_threshold: float = inference_config.LAYOUT_SAME_REGION_THRESHOLD,
    subregion_threshold: float = inference_config.LAYOUT_SUBREGION_THRESHOLD,
) -> List[LayoutElement]:
    """Merge two layouts to produce a single layout."""
    extracted_elements_to_add: List[TextRegion] = []
    inferred_regions_to_remove = []
    w, h = page_image_size
    full_page_region = Rectangle(0, 0, w, h)
    for extracted_region in extracted_layout:
        extracted_is_image = isinstance(extracted_region, ImageTextRegion)
        if extracted_is_image:
            # Skip extracted images for this purpose, we don't have the text from them and they
            # don't provide good text bounding boxes.

            is_full_page_image = region_bounding_boxes_are_almost_the_same(
                extracted_region,
                full_page_region,
                FULL_PAGE_REGION_THRESHOLD,
            )

            if is_full_page_image:
                continue
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
                    if extracted_is_image:
                        # keep extracted region, remove inferred region
                        inferred_regions_to_remove.append(inferred_region)
                    else:
                        # keep inferred region, remove extracted region
                        grow_region_to_match_region(inferred_region, extracted_region)
                        inferred_region.text = extracted_region.text
                        region_matched = True
                elif extracted_is_subregion_of_inferred and inferred_is_text:
                    if extracted_is_image:
                        # keep both extracted and inferred regions
                        region_matched = False
                    else:
                        # keep inferred region, remove extracted region
                        grow_region_to_match_region(inferred_region, extracted_region)
                        region_matched = True
                elif either_region_is_subregion_of_other and inferred_region.type != "Table":
                    # keep extracted region, remove inferred region
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
            source=el.source,
        )
        for el in extracted_elements_to_add
    ]
    inferred_regions_to_add = [
        region for region in inferred_layout if region not in inferred_regions_to_remove
    ]

    final_layout = categorized_extracted_elements_to_add + inferred_regions_to_add

    return final_layout


def separate(region_a: Union[LayoutElement, Rectangle], region_b: Union[LayoutElement, Rectangle]):
    """Reduce leftmost rectangle to don't overlap with the other"""

    def reduce(keep: Rectangle, reduce: Rectangle):
        # Asume intersection

        # Other is down
        if reduce.y2 > keep.y2 and reduce.x1 < keep.x2:
            # other is down-right
            if reduce.x2 > keep.x2 and reduce.y2 > keep.y2:
                reduce.x1 = keep.x2 * 1.01
                reduce.y1 = keep.y2 * 1.01
                return
            # other is down-left
            if reduce.x1 < keep.x1 and reduce.y1 < keep.y2:
                reduce.y1 = keep.y2
                return
            # other is centered
            reduce.y1 = keep.y2
        else:  # other is up
            # other is up-right
            if reduce.x2 > keep.x2 and reduce.y1 < keep.y1:
                reduce.y2 = keep.y1
                return
            # other is left
            if reduce.x1 < keep.x1 and reduce.y1 < keep.y1:
                reduce.y2 = keep.y1
                return
            # other is centered
            reduce.y2 = keep.y1

    if not region_a.intersects(region_b):
        return
    else:
        if region_a.area > region_b.area:
            reduce(keep=region_a, reduce=region_b)
        else:
            reduce(keep=region_b, reduce=region_a)


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


def table_cells_to_dataframe(cells: dict, nrows: int = 1, ncols: int = 1, header=None) -> DataFrame:
    """convert table-transformer's cells data into a pandas dataframe"""
    arr = np.empty((nrows, ncols), dtype=object)
    for cell in cells:
        rows = cell["row_nums"]
        cols = cell["column_nums"]
        if rows[0] >= nrows or cols[0] >= ncols:
            new_arr = np.empty((max(rows[0] + 1, nrows), max(cols[0] + 1, ncols)), dtype=object)
            new_arr[:nrows, :ncols] = arr
            arr = new_arr
            nrows, ncols = arr.shape
        arr[rows[0], cols[0]] = cell["cell text"]

    return DataFrame(arr, columns=header)
