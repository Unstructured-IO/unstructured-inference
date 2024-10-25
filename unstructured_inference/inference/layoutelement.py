from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Iterable, List, Optional

import numpy as np
from layoutparser.elements.layout import TextBlock
from pandas import DataFrame
from scipy.sparse.csgraph import connected_components

from unstructured_inference.config import inference_config
from unstructured_inference.constants import (
    FULL_PAGE_REGION_THRESHOLD,
    ElementType,
    Source,
)
from unstructured_inference.inference.elements import (
    ImageTextRegion,
    Rectangle,
    TextRegion,
    TextRegions,
    coords_intersections,
    grow_region_to_match_region,
    region_bounding_boxes_are_almost_the_same,
)

EPSILON_AREA = 1e-7


@dataclass
class LayoutElements(TextRegions):
    element_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    element_class_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    element_class_id_map: dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        element_size = self.element_coords.shape[0]
        for attr in ("element_probs", "element_class_ids", "texts"):
            if getattr(self, attr).size == 0 and element_size:
                setattr(self, attr, np.array([None] * element_size))

        self.element_probs = self.element_probs.astype(float)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LayoutElements):
            return NotImplemented

        mask = ~np.isnan(self.element_probs)
        other_mask = ~np.isnan(other.element_probs)
        return (
            np.array_equal(self.element_coords, other.element_coords)
            and np.array_equal(self.texts, other.texts)
            and np.array_equal(mask, other_mask)
            and np.array_equal(self.element_probs[mask], other.element_probs[mask])
            and (
                [self.element_class_id_map[idx] for idx in self.element_class_ids]
                == [other.element_class_id_map[idx] for idx in other.element_class_ids]
            )
            and self.source == other.source
        )

    def slice(self, indices) -> LayoutElements:
        """slice and return only selected indices"""
        return LayoutElements(
            element_coords=self.element_coords[indices],
            texts=self.texts[indices],
            source=self.source,
            element_probs=self.element_probs[indices],
            element_class_ids=self.element_class_ids[indices],
            element_class_id_map=self.element_class_id_map,
        )

    @classmethod
    def concatenate(cls, groups: Iterable[LayoutElements]) -> LayoutElements:
        """concatenate a sequence of LayoutElements in order as one LayoutElements"""
        coords, texts, probs, class_ids, sources = [], [], [], [], []
        class_id_map = {}
        for group in groups:
            coords.append(group.element_coords)
            texts.append(group.texts)
            probs.append(group.element_probs)
            class_ids.append(group.element_class_ids)
            if group.source:
                sources.append(group.source)
            if group.element_class_id_map:
                class_id_map.update(group.element_class_id_map)
        return cls(
            element_coords=np.concatenate(coords),
            texts=np.concatenate(texts),
            element_probs=np.concatenate(probs),
            element_class_ids=np.concatenate(class_ids),
            element_class_id_map=class_id_map,
            source=sources[0] if sources else None,
        )

    def as_list(self):
        """return a list of LayoutElement for backward compatibility"""
        return [
            LayoutElement.from_coords(
                x1,
                y1,
                x2,
                y2,
                text=text,
                type=(
                    self.element_class_id_map[class_id]
                    if class_id is not None and self.element_class_id_map
                    else None
                ),
                prob=None if np.isnan(prob) else prob,
                source=self.source,
            )
            for (x1, y1, x2, y2), text, prob, class_id in zip(
                self.element_coords,
                self.texts,
                self.element_probs,
                self.element_class_ids,
            )
        ]

    @classmethod
    def from_list(cls, elements: list):
        """create LayoutElements from a list of LayoutElement objects; the objects must have the
        same source"""
        len_ele = len(elements)
        coords = np.empty((len_ele, 4), dtype=float)
        # text and probs can be Nones so use lists first then convert into array to avoid them being
        # filled as nan
        texts = []
        class_probs = []
        class_types = np.empty((len_ele,), dtype="object")

        for i, element in enumerate(elements):
            coords[i] = [element.bbox.x1, element.bbox.y1, element.bbox.x2, element.bbox.y2]
            texts.append(element.text)
            class_probs.append(element.prob)
            class_types[i] = element.type or "None"

        unique_ids, class_ids = np.unique(class_types, return_inverse=True)
        unique_ids[unique_ids == "None"] = None

        return cls(
            element_coords=coords,
            texts=np.array(texts),
            element_probs=np.array(class_probs),
            element_class_ids=class_ids,
            element_class_id_map=dict(zip(range(len(unique_ids)), unique_ids)),
            source=elements[0].source if len_ele else None,
        )


@dataclass
class LayoutElement(TextRegion):
    type: Optional[str] = None
    prob: Optional[float] = None
    image_path: Optional[str] = None
    parent: Optional[LayoutElement] = None

    def to_dict(self) -> dict:
        """Converts the class instance to dictionary form."""
        out_dict = {
            "coordinates": None if self.bbox is None else self.bbox.coordinates,
            "text": self.text,
            "type": self.type,
            "prob": self.prob,
            "source": self.source,
        }
        return out_dict

    @classmethod
    def from_region(cls, region: TextRegion):
        """Create LayoutElement from superclass."""
        text = region.text if hasattr(region, "text") else None
        type = region.type if hasattr(region, "type") else None
        prob = region.prob if hasattr(region, "prob") else None
        source = region.source if hasattr(region, "source") else None
        return cls(text=text, source=source, type=type, prob=prob, bbox=region.bbox)

    @classmethod
    def from_lp_textblock(cls, textblock: TextBlock):
        """Create LayoutElement from layoutparser TextBlock object."""
        x1, y1, x2, y2 = textblock.coordinates
        text = textblock.text
        type = textblock.type
        prob = textblock.score
        return cls.from_coords(
            x1,
            y1,
            x2,
            y2,
            text=text,
            source=Source.DETECTRON2_LP,
            type=type,
            prob=prob,
        )


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
                extracted_region.bbox,
                full_page_region,
                FULL_PAGE_REGION_THRESHOLD,
            )

            if is_full_page_image:
                continue
        region_matched = False
        for inferred_region in inferred_layout:

            if inferred_region.bbox.intersects(extracted_region.bbox):
                same_bbox = region_bounding_boxes_are_almost_the_same(
                    inferred_region.bbox,
                    extracted_region.bbox,
                    same_region_threshold,
                )
                inferred_is_subregion_of_extracted = inferred_region.bbox.is_almost_subregion_of(
                    extracted_region.bbox,
                    subregion_threshold=subregion_threshold,
                )
                inferred_is_text = inferred_region.type not in (
                    ElementType.FIGURE,
                    ElementType.IMAGE,
                    ElementType.PAGE_BREAK,
                    ElementType.TABLE,
                )
                extracted_is_subregion_of_inferred = extracted_region.bbox.is_almost_subregion_of(
                    inferred_region.bbox,
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
                        grow_region_to_match_region(inferred_region.bbox, extracted_region.bbox)
                        inferred_region.text = extracted_region.text
                        region_matched = True
                elif extracted_is_subregion_of_inferred and inferred_is_text:
                    if extracted_is_image:
                        # keep both extracted and inferred regions
                        region_matched = False
                    else:
                        # keep inferred region, remove extracted region
                        grow_region_to_match_region(inferred_region.bbox, extracted_region.bbox)
                        region_matched = True
                elif (
                    either_region_is_subregion_of_other
                    and inferred_region.type != ElementType.TABLE
                ):
                    # keep extracted region, remove inferred region
                    inferred_regions_to_remove.append(inferred_region)
        if not region_matched:
            extracted_elements_to_add.append(extracted_region)
    # Need to classify the extracted layout elements we're keeping.
    categorized_extracted_elements_to_add = [
        LayoutElement(
            text=el.text,
            type=(
                ElementType.IMAGE
                if isinstance(el, ImageTextRegion)
                else ElementType.UNCATEGORIZED_TEXT
            ),
            source=el.source,
            bbox=el.bbox,
        )
        for el in extracted_elements_to_add
    ]
    inferred_regions_to_add = [
        region for region in inferred_layout if region not in inferred_regions_to_remove
    ]

    final_layout = categorized_extracted_elements_to_add + inferred_regions_to_add

    return final_layout


def separate(region_a: Rectangle, region_b: Rectangle):
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


def table_cells_to_dataframe(
    cells: List[dict],
    nrows: int = 1,
    ncols: int = 1,
    header=None,
) -> DataFrame:
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


def partition_groups_from_regions(regions: TextRegions) -> List[TextRegions]:
    """Partitions regions into groups of regions based on proximity. Returns list of lists of
    regions, each list corresponding with a group"""
    if len(regions) == 0:
        return []
    padded_coords = regions.element_coords.copy().astype(float)
    v_pad = (regions.y2 - regions.y1) * inference_config.ELEMENTS_V_PADDING_COEF
    h_pad = (regions.x2 - regions.x1) * inference_config.ELEMENTS_H_PADDING_COEF
    padded_coords[:, 0] -= h_pad
    padded_coords[:, 1] -= v_pad
    padded_coords[:, 2] += h_pad
    padded_coords[:, 3] += v_pad

    intersection_mtx = coords_intersections(padded_coords)

    group_count, group_nums = connected_components(intersection_mtx)
    groups: List[TextRegions] = []
    for group in range(group_count):
        groups.append(regions.slice(np.where(group_nums == group)[0]))

    return groups


def intersection_areas_between_coords(
    coords1: np.ndarray,
    coords2: np.ndarray,
    threshold: float = 0.5,
):
    """compute intersection area and own areas for two groups of bounding boxes"""
    x11, y11, x12, y12 = np.split(coords1, 4, axis=1)
    x21, y21, x22, y22 = np.split(coords2, 4, axis=1)

    xa = np.maximum(x11, np.transpose(x21))
    ya = np.maximum(y11, np.transpose(y21))
    xb = np.minimum(x12, np.transpose(x22))
    yb = np.minimum(y12, np.transpose(y22))

    return np.maximum((xb - xa), 0) * np.maximum((yb - ya), 0)


def clean_layoutelements(elements: LayoutElements, subregion_threshold: float = 0.5):
    """After this function, the list of elements will not contain any element inside
    of the type specified"""
    # Sort elements from biggest to smallest
    if len(elements) < 2:
        return elements

    sorted_by_area = np.argsort(-elements.areas)
    sorted_coords = elements.element_coords[sorted_by_area]

    # First check if targets contains each other
    self_intersection = intersection_areas_between_coords(sorted_coords, sorted_coords)
    areas = elements.areas[sorted_by_area]
    # check from largest to smallest regions to find if it contains any other regions
    is_almost_subregion_of = (
        self_intersection / np.maximum(areas, EPSILON_AREA) > subregion_threshold
    ) & (areas <= areas.T)

    n_candidates = len(elements)
    mask = np.ones_like(areas, dtype=bool)
    current_candidate = 0
    while n_candidates > 1:
        plus_one = current_candidate + 1
        remove = (
            np.where(is_almost_subregion_of[current_candidate, plus_one:])[0]
            + current_candidate
            + 1
        )

        if not remove.sum():
            break

        mask[remove] = 0
        n_candidates -= len(remove) + 1
        remaining_candidates = np.where(mask[plus_one:])[0]

        if not len(remaining_candidates):
            break

        current_candidate = remaining_candidates[0] + plus_one

    final_coords = sorted_coords[mask]
    sorted_by_y1 = np.argsort(final_coords[:, 1])

    final_attrs: dict[str, Any] = {
        "element_class_id_map": elements.element_class_id_map,
        "source": elements.source,
    }
    for attr in ("element_class_ids", "element_probs", "texts"):
        if (original_attr := getattr(elements, attr)) is None:
            continue
        final_attrs[attr] = original_attr[sorted_by_area][mask][sorted_by_y1]
    final_elements = LayoutElements(element_coords=final_coords[sorted_by_y1], **final_attrs)
    return final_elements


def clean_layoutelements_for_class(
    elements: LayoutElements,
    element_class: int,
    subregion_threshold: float = 0.5,
):
    """After this function, the list of elements will not contain any element inside
    of the type specified"""
    # Sort elements from biggest to smallest
    sorted_by_area = np.argsort(-elements.areas)
    sorted_coords = elements.element_coords[sorted_by_area]

    target_indices = elements.element_class_ids[sorted_by_area] == element_class

    # skip trivial result
    len_target = target_indices.sum()
    if len_target == 0 or len_target == len(elements):
        return elements

    target_coords = sorted_coords[target_indices]
    other_coords = sorted_coords[~target_indices]

    # First check if targets contains each other
    target_self_intersection = intersection_areas_between_coords(target_coords, target_coords)
    target_areas = elements.areas[sorted_by_area][target_indices]
    # check from largest to smallest regions to find if it contains any other regions
    is_almost_subregion_of = (
        target_self_intersection / np.maximum(target_areas, EPSILON_AREA) > subregion_threshold
    ) & (target_areas <= target_areas.T)

    n_candidates = len_target
    mask = np.ones_like(target_areas, dtype=bool)
    current_candidate = 0
    while n_candidates > 1:
        plus_one = current_candidate + 1
        remove = (
            np.where(is_almost_subregion_of[current_candidate, plus_one:])[0]
            + current_candidate
            + 1
        )

        if not remove.sum():
            break

        mask[remove] = 0
        n_candidates -= len(remove) + 1
        remaining_candidates = np.where(mask[plus_one:])[0]

        if not len(remaining_candidates):
            break

        current_candidate = remaining_candidates[0] + plus_one

    target_coords_to_keep = target_coords[mask]

    other_to_target_intersection = intersection_areas_between_coords(
        other_coords,
        target_coords_to_keep,
    )
    # check from largest to smallest regions to find if it contains any other regions
    other_areas = elements.areas[sorted_by_area][~target_indices]
    other_is_almost_subregion_of_target = (
        other_to_target_intersection / np.maximum(other_areas, EPSILON_AREA) > subregion_threshold
    ) & (other_areas.reshape((-1, 1)) <= target_areas[mask].T)

    other_mask = ~other_is_almost_subregion_of_target.sum(axis=1).astype(bool)

    final_coords = np.vstack([target_coords[mask], other_coords[other_mask]])
    final_attrs: dict[str, Any] = {"element_class_id_map": elements.element_class_id_map}
    for attr in ("element_class_ids", "element_probs", "texts"):
        if (original_attr := getattr(elements, attr)) is None:
            continue
        final_attrs[attr] = np.concatenate(
            (original_attr[target_indices][mask], original_attr[~target_indices][other_mask]),
        )
    final_elements = LayoutElements(element_coords=final_coords, **final_attrs)
    return final_elements
