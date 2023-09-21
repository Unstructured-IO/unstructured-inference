from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, List, Optional, cast

import numpy as np
from layoutparser.elements.layout import TextBlock
from pandas import DataFrame
from PIL import Image

from unstructured_inference.constants import FULL_PAGE_REGION_THRESHOLD, SUBREGION_THRESHOLD_FOR_OCR
from unstructured_inference.inference.elements import (
    ImageTextRegion,
    Rectangle,
    TextRegion,
    grow_region_to_match_region,
    partition_groups_from_regions,
    region_bounding_boxes_are_almost_the_same,
)
from unstructured_inference.models import tables


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
            "prob": self.prob,
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
        score = textblock.score
        return cls(x1, y1, x2, y2, text, type, prob=score)


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
    page_image_size: tuple,
    ocr_layout: Optional[List[TextRegion]] = None,
    supplement_with_ocr_elements: bool = True,
    same_region_threshold: float = 0.75,
    subregion_threshold: float = 0.75,
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
    inferred_regions_to_add = [
        region for region in inferred_layout if region not in inferred_regions_to_remove
    ]
    inferred_regions_to_add_without_text = [
        region for region in inferred_regions_to_add if not region.text
    ]
    if ocr_layout is not None:
        for inferred_region in inferred_regions_to_add_without_text:
            inferred_region.text = aggregate_ocr_text_by_block(
                ocr_layout,
                inferred_region,
                SUBREGION_THRESHOLD_FOR_OCR,
            )
        out_layout = categorized_extracted_elements_to_add + inferred_regions_to_add
        final_layout = (
            supplement_layout_with_ocr_elements(out_layout, ocr_layout)
            if supplement_with_ocr_elements
            else out_layout
        )
    else:
        final_layout = categorized_extracted_elements_to_add + inferred_regions_to_add

    return final_layout


def merge_inferred_layout_with_ocr_layout(
    inferred_layout: List[LayoutElement],
    ocr_layout: List[TextRegion],
    supplement_with_ocr_elements: bool = True,
) -> List[LayoutElement]:
    """
    Merge the inferred layout with the OCR-detected text regions.

    This function iterates over each inferred layout element and aggregates the
    associated text from the OCR layout using the specified threshold. The inferred
    layout's text attribute is then updated with this aggregated text.
    """

    for inferred_region in inferred_layout:
        inferred_region.text = aggregate_ocr_text_by_block(
            ocr_layout,
            inferred_region,
            SUBREGION_THRESHOLD_FOR_OCR,
        )

    final_layout = (
        supplement_layout_with_ocr_elements(inferred_layout, ocr_layout)
        if supplement_with_ocr_elements
        else inferred_layout
    )

    return final_layout


def aggregate_ocr_text_by_block(
    ocr_layout: List[TextRegion],
    region: TextRegion,
    subregion_threshold: float,
) -> Optional[str]:
    """Extracts the text aggregated from the regions of the ocr layout that lie within the given
    block."""

    extracted_texts = []

    for ocr_region in ocr_layout:
        ocr_region_is_subregion_of_given_region = ocr_region.is_almost_subregion_of(
            region,
            subregion_threshold=subregion_threshold,
        )
        if ocr_region_is_subregion_of_given_region and ocr_region.text:
            extracted_texts.append(ocr_region.text)

    return " ".join(extracted_texts) if extracted_texts else None


def supplement_layout_with_ocr_elements(
    layout: List[LayoutElement],
    ocr_layout: List[TextRegion],
) -> List[LayoutElement]:
    """
    Supplement the existing layout with additional OCR-derived elements.

    This function takes two lists: one list of pre-existing layout elements (`layout`)
    and another list of OCR-detected text regions (`ocr_layout`). It identifies OCR regions
    that are subregions of the elements in the existing layout and removes them from the
    OCR-derived list. Then, it appends the remaining OCR-derived regions to the existing layout.

    Parameters:
    - layout (List[LayoutElement]): A list of existing layout elements, each of which is
                                    an instance of `LayoutElement`.
    - ocr_layout (List[TextRegion]): A list of OCR-derived text regions, each of which is
                                     an instance of `TextRegion`.

    Returns:
    - List[LayoutElement]: The final combined layout consisting of both the original layout
                           elements and the new OCR-derived elements.

    Note:
    - The function relies on `is_almost_subregion_of()` method to determine if an OCR region
      is a subregion of an existing layout element.
    - It also relies on `get_elements_from_ocr_regions()` to convert OCR regions to layout elements.
    - The `SUBREGION_THRESHOLD_FOR_OCR` constant is used to specify the subregion matching
     threshold.
    """

    ocr_regions_to_remove = []
    for ocr_region in ocr_layout:
        for el in layout:
            ocr_region_is_subregion_of_out_el = ocr_region.is_almost_subregion_of(
                cast(Rectangle, el),
                SUBREGION_THRESHOLD_FOR_OCR,
            )
            if ocr_region_is_subregion_of_out_el:
                ocr_regions_to_remove.append(ocr_region)
                break

    ocr_regions_to_add = [region for region in ocr_layout if region not in ocr_regions_to_remove]
    if ocr_regions_to_add:
        ocr_elements_to_add = get_elements_from_ocr_regions(ocr_regions_to_add)
        final_layout = layout + ocr_elements_to_add
    else:
        final_layout = layout

    return final_layout


def merge_text_regions(regions: List[TextRegion]) -> TextRegion:
    """
    Merge a list of TextRegion objects into a single TextRegion.

    Parameters:
    - group (List[TextRegion]): A list of TextRegion objects to be merged.

    Returns:
    - TextRegion: A single merged TextRegion object.
    """

    min_x1 = min([tr.x1 for tr in regions])
    min_y1 = min([tr.y1 for tr in regions])
    max_x2 = max([tr.x2 for tr in regions])
    max_y2 = max([tr.y2 for tr in regions])

    merged_text = " ".join([tr.text for tr in regions if tr.text])

    return TextRegion(min_x1, min_y1, max_x2, max_y2, merged_text)


def get_elements_from_ocr_regions(ocr_regions: List[TextRegion]) -> List[LayoutElement]:
    """
    Get layout elements from OCR regions
    """

    grouped_regions = cast(
        List[List[TextRegion]],
        partition_groups_from_regions(ocr_regions),
    )
    merged_regions = [merge_text_regions(group) for group in grouped_regions]
    return [
        LayoutElement(
            r.x1,
            r.y1,
            r.x2,
            r.y2,
            text=r.text,
            type="UncategorizedText",
        )
        for r in merged_regions
    ]


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
