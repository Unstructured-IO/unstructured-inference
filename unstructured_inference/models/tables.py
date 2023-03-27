# https://github.com/microsoft/table-transformer/blob/main/src/inference.py
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Table%20Transformer/Using_Table_Transformer_for_table_detection_and_table_structure_recognition.ipynb
import torch
import logging

from unstructured_inference.models.unstructuredmodel import UnstructuredModel
from unstructured_inference.logger import logger

from collections import defaultdict
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd

import pytesseract

from transformers import TableTransformerForObjectDetection
from transformers import DetrImageProcessor
from PIL import Image
from typing import Union, Optional
from pathlib import Path
import platform

from . import table_postprocess as postprocess
from unstructured_inference.models.table_postprocess import Rect


class UnstructuredTableTransformerModel(UnstructuredModel):
    """Unstructured model wrapper for table-transformer."""

    def __init__(self):
        pass

    def predict(self, x: Image):
        """Predict table structure deferring to run_prediction"""
        super().predict(x)
        return self.run_prediction(x)

    def initialize(
        self,
        model: Union[str, Path, TableTransformerForObjectDetection] = None,
        device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Loads the donut model using the specified parameters"""
        self.device = device
        self.feature_extractor = DetrImageProcessor()

        try:
            logging.info("Loading the table structure model ...")
            self.model = TableTransformerForObjectDetection.from_pretrained(model)
            self.model.eval()

        except EnvironmentError:
            logging.critical("Failed to initialize the model.")
            logging.critical("Ensure that the model is correct")
            raise ImportError(
                "Review the parameters to initialize a UnstructuredTableTransformerModel obj"
            )
        self.model.to(device)

    def run_prediction(self, x: Image):
        """Predict table structure"""
        with torch.no_grad():
            encoding = self.feature_extractor(x, return_tensors="pt").to(self.device)
            outputs_structure = self.model(**encoding)

        if platform.machine() == "x86_64":
            from unstructured_inference.models import paddle_ocr

            paddle_result = paddle_ocr.load_agent().ocr(np.array(x), cls=True)

            tokens = []
            for idx in range(len(paddle_result)):
                res = paddle_result[idx]
                for line in res:
                    xmin = min([i[0] for i in line[0]])
                    ymin = min([i[1] for i in line[0]])
                    xmax = max([i[0] for i in line[0]])
                    ymax = max([i[1] for i in line[0]])
                    tokens.append({"bbox": [xmin, ymin, xmax, ymax], "text": line[1][0]})
        else:
            zoom = 6
            img = cv2.resize(
                cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR),
                None,
                fx=zoom,
                fy=zoom,
                interpolation=cv2.INTER_CUBIC,
            )

            kernel = np.ones((1, 1), np.uint8)
            img = cv2.dilate(img, kernel, iterations=1)
            img = cv2.erode(img, kernel, iterations=1)

            ocr_df: pd.DataFrame = pytesseract.image_to_data(
                Image.fromarray(img), output_type="data.frame"
            )

            ocr_df = ocr_df.dropna()

            tokens = []
            for idtx in ocr_df.itertuples():
                tokens.append(
                    {
                        "bbox": [
                            idtx.left / zoom,
                            idtx.top / zoom,
                            (idtx.left + idtx.width) / zoom,
                            (idtx.top + idtx.height) / zoom,
                        ],
                        "text": idtx.text,
                    }
                )

        sorted(tokens, key=lambda x: x["bbox"][1] * 10000 + x["bbox"][0])

        # 'tokens' is a list of tokens
        # Need to be in a relative reading order
        # If no order is provided, use current order
        for idx, token in enumerate(tokens):
            if "span_num" not in token:
                token["span_num"] = idx
            if "line_num" not in token:
                token["line_num"] = 0
            if "block_num" not in token:
                token["block_num"] = 0

        html = recognize(outputs_structure, x, tokens=tokens, out_html=True)["html"]
        prediction = html[0] if html else ""
        return prediction


tables_agent: UnstructuredTableTransformerModel = UnstructuredTableTransformerModel()


def load_agent():
    """Loads the Tesseract OCR agent as a global variable to ensure that we only load it once."""
    global tables_agent

    if not hasattr(tables_agent, "model"):
        logger.info("Loading the Tesseract OCR agent ...")
        tables_agent.initialize("microsoft/table-transformer-structure-recognition")

    return


def get_class_map(data_type: str):
    """Defines class map dictionaries"""
    if data_type == "structure":
        class_map = {
            "table": 0,
            "table column": 1,
            "table row": 2,
            "table column header": 3,
            "table projected row header": 4,
            "table spanning cell": 5,
            "no object": 6,
        }
    elif data_type == "detection":
        class_map = {"table": 0, "table rotated": 1, "no object": 2}
    return class_map


structure_class_thresholds = {
    "table": 0.5,
    "table column": 0.5,
    "table row": 0.5,
    "table column header": 0.5,
    "table projected row header": 0.5,
    "table spanning cell": 0.5,
    "no object": 10,
}


def recognize(outputs: dict, img: Image, tokens: list, out_html: bool = False):
    """Recognize table elements."""
    out_formats = {}

    str_class_name2idx = get_class_map("structure")
    str_class_idx2name = {v: k for k, v in str_class_name2idx.items()}
    str_class_thresholds = structure_class_thresholds

    # Post-process detected objects, assign class labels
    objects = outputs_to_objects(outputs, img.size, str_class_idx2name)

    # Further process the detected objects so they correspond to a consistent table
    tables_structure = objects_to_structures(objects, tokens, str_class_thresholds)
    # Enumerate all table cells: grid cells and spanning cells
    tables_cells = [structure_to_cells(structure, tokens)[0] for structure in tables_structure]

    # Convert cells to HTML
    if out_html:
        tables_htmls = [cells_to_html(cells) for cells in tables_cells]
        out_formats["html"] = tables_htmls

    return out_formats


def outputs_to_objects(outputs, img_size, class_idx2name):
    """Output table element types."""
    m = outputs["logits"].softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    """Convert rectangle format from center-x, center-y, width, height to
    x-min, y-min, x-max, y-max."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    """Rescale relative bounding box to box of size given by size."""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(Rect(bbox2))

    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area

    return 0


def objects_to_structures(objects, tokens, class_thresholds):
    """
    Process the bounding boxes produced by the table structure recognition model into
    a *consistent* set of table structures (rows, columns, spanning cells, headers).
    This entails resolving conflicts/overlaps, and ensuring the boxes meet certain alignment
    conditions (for example: rows should all have the same width, etc.).
    """

    tables = [obj for obj in objects if obj["label"] == "table"]
    table_structures = []

    for table in tables:
        table_objects = [obj for obj in objects if iob(obj["bbox"], table["bbox"]) >= 0.5]
        table_tokens = [token for token in tokens if iob(token["bbox"], table["bbox"]) >= 0.5]

        structure = {}

        columns = [obj for obj in table_objects if obj["label"] == "table column"]
        rows = [obj for obj in table_objects if obj["label"] == "table row"]
        column_headers = [obj for obj in table_objects if obj["label"] == "table column header"]
        spanning_cells = [obj for obj in table_objects if obj["label"] == "table spanning cell"]
        for obj in spanning_cells:
            obj["projected row header"] = False
        projected_row_headers = [
            obj for obj in table_objects if obj["label"] == "table projected row header"
        ]
        for obj in projected_row_headers:
            obj["projected row header"] = True
        spanning_cells += projected_row_headers
        for obj in rows:
            obj["column header"] = False
            for header_obj in column_headers:
                if iob(obj["bbox"], header_obj["bbox"]) >= 0.5:
                    obj["column header"] = True

        # Refine table structures
        rows = postprocess.refine_rows(rows, table_tokens, class_thresholds["table row"])
        columns = postprocess.refine_columns(
            columns, table_tokens, class_thresholds["table column"]
        )

        # Shrink table bbox to just the total height of the rows
        # and the total width of the columns
        row_rect = Rect()
        for obj in rows:
            row_rect.include_rect(obj["bbox"])
        column_rect = Rect()
        for obj in columns:
            column_rect.include_rect(obj["bbox"])
        table["row_column_bbox"] = [
            column_rect.x_min,
            row_rect.y_min,
            column_rect.x_max,
            row_rect.y_max,
        ]
        table["bbox"] = table["row_column_bbox"]

        # Process the rows and columns into a complete segmented table
        columns = postprocess.align_columns(columns, table["row_column_bbox"])
        rows = postprocess.align_rows(rows, table["row_column_bbox"])

        structure["rows"] = rows
        structure["columns"] = columns
        structure["column headers"] = column_headers
        structure["spanning cells"] = spanning_cells

        if len(rows) > 0 and len(columns) > 1:
            structure = refine_table_structure(structure, class_thresholds)

        table_structures.append(structure)

    return table_structures


def refine_table_structure(table_structure, class_thresholds):
    """
    Apply operations to the detected table structure objects such as
    thresholding, NMS, and alignment.
    """
    rows = table_structure["rows"]
    columns = table_structure["columns"]

    # Process the headers
    column_headers = table_structure["column headers"]
    column_headers = postprocess.apply_threshold(
        column_headers, class_thresholds["table column header"]
    )
    column_headers = postprocess.nms(column_headers)
    column_headers = align_headers(column_headers, rows)

    # Process spanning cells
    spanning_cells = [
        elem for elem in table_structure["spanning cells"] if not elem["projected row header"]
    ]
    projected_row_headers = [
        elem for elem in table_structure["spanning cells"] if elem["projected row header"]
    ]
    spanning_cells = postprocess.apply_threshold(
        spanning_cells, class_thresholds["table spanning cell"]
    )
    projected_row_headers = postprocess.apply_threshold(
        projected_row_headers, class_thresholds["table projected row header"]
    )
    spanning_cells += projected_row_headers
    # Align before NMS for spanning cells because alignment brings them into agreement
    # with rows and columns first; if spanning cells still overlap after this operation,
    # the threshold for NMS can basically be lowered to just above 0
    spanning_cells = postprocess.align_supercells(spanning_cells, rows, columns)
    spanning_cells = postprocess.nms_supercells(spanning_cells)

    postprocess.header_supercell_tree(spanning_cells)

    table_structure["columns"] = columns
    table_structure["rows"] = rows
    table_structure["spanning cells"] = spanning_cells
    table_structure["column headers"] = column_headers

    return table_structure


def align_headers(headers, rows):
    """
    Adjust the header boundary to be the convex hull of the rows it intersects
    at least 50% of the height of.

    For now, we are not supporting tables with multiple headers, so we need to
    eliminate anything besides the top-most header.
    """

    aligned_headers = []

    for row in rows:
        row["column header"] = False

    header_row_nums = []
    for header in headers:
        for row_num, row in enumerate(rows):
            row_height = row["bbox"][3] - row["bbox"][1]
            min_row_overlap = max(row["bbox"][1], header["bbox"][1])
            max_row_overlap = min(row["bbox"][3], header["bbox"][3])
            overlap_height = max_row_overlap - min_row_overlap
            if overlap_height / row_height >= 0.5:
                header_row_nums.append(row_num)

    if len(header_row_nums) == 0:
        return aligned_headers

    header_rect = Rect()
    if header_row_nums[0] > 0:
        header_row_nums = list(range(header_row_nums[0] + 1)) + header_row_nums

    last_row_num = -1
    for row_num in header_row_nums:
        if row_num == last_row_num + 1:
            row = rows[row_num]
            row["column header"] = True
            header_rect = header_rect.include_rect(row["bbox"])
            last_row_num = row_num
        else:
            # Break as soon as a non-header row is encountered.
            # This ignores any subsequent rows in the table labeled as a header.
            # Having more than 1 header is not supported currently.
            break

    header = {"bbox": header_rect.get_bbox()}
    aligned_headers.append(header)

    return aligned_headers


def structure_to_cells(table_structure, tokens):
    """
    Assuming the row, column, spanning cell, and header bounding boxes have
    been refined into a set of consistent table structures, process these
    table structures into table cells. This is a universal representation
    format for the table, which can later be exported to Pandas or CSV formats.
    Classify the cells as header/access cells or data cells
    based on if they intersect with the header bounding box.
    """
    columns = table_structure["columns"]
    rows = table_structure["rows"]
    spanning_cells = table_structure["spanning cells"]
    cells = []
    subcells = []
    # Identify complete cells and subcells
    for column_num, column in enumerate(columns):
        for row_num, row in enumerate(rows):
            column_rect = Rect(list(column["bbox"]))
            row_rect = Rect(list(row["bbox"]))
            cell_rect = row_rect.intersect(column_rect)
            header = "column header" in row and row["column header"]
            cell = {
                "bbox": cell_rect.get_bbox(),
                "column_nums": [column_num],
                "row_nums": [row_num],
                "column header": header,
            }

            cell["subcell"] = False
            for spanning_cell in spanning_cells:
                spanning_cell_rect = Rect(list(spanning_cell["bbox"]))
                if (
                    spanning_cell_rect.intersect(cell_rect).get_area() / cell_rect.get_area()
                ) > 0.5:
                    cell["subcell"] = True
                    break

            if cell["subcell"]:
                subcells.append(cell)
            else:
                # cell text = extract_text_inside_bbox(table_spans, cell['bbox'])
                # cell['cell text'] = cell text
                cell["projected row header"] = False
                cells.append(cell)

    for spanning_cell in spanning_cells:
        spanning_cell_rect = Rect(list(spanning_cell["bbox"]))
        cell_columns = set()
        cell_rows = set()
        cell_rect = None
        header = True
        for subcell in subcells:
            subcell_rect = Rect(list(subcell["bbox"]))
            subcell_rect_area = subcell_rect.get_area()
            if (subcell_rect.intersect(spanning_cell_rect).get_area() / subcell_rect_area) > 0.5:
                if cell_rect is None:
                    cell_rect = Rect(list(subcell["bbox"]))
                else:
                    cell_rect.include_rect(list(subcell["bbox"]))
                cell_rows = cell_rows.union(set(subcell["row_nums"]))
                cell_columns = cell_columns.union(set(subcell["column_nums"]))
                # By convention here, all subcells must be classified
                # as header cells for a spanning cell to be classified as a header cell;
                # otherwise, this could lead to a non-rectangular header region
                header = header and "column header" in subcell and subcell["column header"]
        if len(cell_rows) > 0 and len(cell_columns) > 0:
            cell = {
                "bbox": cell_rect.get_bbox(),
                "column_nums": list(cell_columns),
                "row_nums": list(cell_rows),
                "column header": header,
                "projected row header": spanning_cell["projected row header"],
            }
            cells.append(cell)

    # Compute a confidence score based on how well the page tokens
    # slot into the cells reported by the model
    _, _, cell_match_scores = postprocess.slot_into_containers(cells, tokens)
    try:
        mean_match_score = sum(cell_match_scores) / len(cell_match_scores)
        min_match_score = min(cell_match_scores)
        confidence_score = (mean_match_score + min_match_score) / 2
    except ZeroDivisionError:
        confidence_score = 0

    # Dilate rows and columns before final extraction
    # dilated_columns = fill_column_gaps(columns, table_bbox)
    dilated_columns = columns
    # dilated_rows = fill_row_gaps(rows, table_bbox)
    dilated_rows = rows
    for cell in cells:
        column_rect = Rect()
        for column_num in cell["column_nums"]:
            column_rect.include_rect(list(dilated_columns[column_num]["bbox"]))
        row_rect = Rect()
        for row_num in cell["row_nums"]:
            row_rect.include_rect(list(dilated_rows[row_num]["bbox"]))
        cell_rect = column_rect.intersect(row_rect)
        cell["bbox"] = cell_rect.get_bbox()

    span_nums_by_cell, _, _ = postprocess.slot_into_containers(
        cells, tokens, overlap_threshold=0.001, forced_assignment=False
    )

    for cell, cell_span_nums in zip(cells, span_nums_by_cell):
        cell_spans = [tokens[num] for num in cell_span_nums]
        # TODO: Refine how text is extracted; should be character-based, not span-based;
        # but need to associate
        cell["cell text"] = postprocess.extract_text_from_spans(
            cell_spans, remove_integer_superscripts=False
        )
        cell["spans"] = cell_spans

    # Adjust the row, column, and cell bounding boxes to reflect the extracted text
    num_rows = len(rows)
    rows = postprocess.sort_objects_top_to_bottom(rows)
    num_columns = len(columns)
    columns = postprocess.sort_objects_left_to_right(columns)
    min_y_values_by_row = defaultdict(list)
    max_y_values_by_row = defaultdict(list)
    min_x_values_by_column = defaultdict(list)
    max_x_values_by_column = defaultdict(list)
    for cell in cells:
        min_row = min(cell["row_nums"])
        max_row = max(cell["row_nums"])
        min_column = min(cell["column_nums"])
        max_column = max(cell["column_nums"])
        for span in cell["spans"]:
            min_x_values_by_column[min_column].append(span["bbox"][0])
            min_y_values_by_row[min_row].append(span["bbox"][1])
            max_x_values_by_column[max_column].append(span["bbox"][2])
            max_y_values_by_row[max_row].append(span["bbox"][3])
    for row_num, row in enumerate(rows):
        if len(min_x_values_by_column[0]) > 0:
            row["bbox"][0] = min(min_x_values_by_column[0])
        if len(min_y_values_by_row[row_num]) > 0:
            row["bbox"][1] = min(min_y_values_by_row[row_num])
        if len(max_x_values_by_column[num_columns - 1]) > 0:
            row["bbox"][2] = max(max_x_values_by_column[num_columns - 1])
        if len(max_y_values_by_row[row_num]) > 0:
            row["bbox"][3] = max(max_y_values_by_row[row_num])
    for column_num, column in enumerate(columns):
        if len(min_x_values_by_column[column_num]) > 0:
            column["bbox"][0] = min(min_x_values_by_column[column_num])
        if len(min_y_values_by_row[0]) > 0:
            column["bbox"][1] = min(min_y_values_by_row[0])
        if len(max_x_values_by_column[column_num]) > 0:
            column["bbox"][2] = max(max_x_values_by_column[column_num])
        if len(max_y_values_by_row[num_rows - 1]) > 0:
            column["bbox"][3] = max(max_y_values_by_row[num_rows - 1])
    for cell in cells:
        row_rect = None
        column_rect = None
        for row_num in cell["row_nums"]:
            if row_rect is None:
                row_rect = Rect(list(rows[row_num]["bbox"]))
            else:
                row_rect.include_rect(list(rows[row_num]["bbox"]))
        for column_num in cell["column_nums"]:
            if column_rect is None:
                column_rect = Rect(list(columns[column_num]["bbox"]))
            else:
                column_rect.include_rect(list(columns[column_num]["bbox"]))
        cell_rect = row_rect.intersect(column_rect)
        if cell_rect.get_area() > 0:
            cell["bbox"] = cell_rect.get_bbox()
            pass

    return cells, confidence_score


def cells_to_html(cells):
    """Convert table structure to html format."""
    cells = sorted(cells, key=lambda k: min(k["column_nums"]))
    cells = sorted(cells, key=lambda k: min(k["row_nums"]))

    table = ET.Element("table")
    current_row = -1

    for cell in cells:
        this_row = min(cell["row_nums"])

        attrib = {}
        colspan = len(cell["column_nums"])
        if colspan > 1:
            attrib["colspan"] = str(colspan)
        rowspan = len(cell["row_nums"])
        if rowspan > 1:
            attrib["rowspan"] = str(rowspan)
        if this_row > current_row:
            current_row = this_row
            if cell["column header"]:
                cell_tag = "th"
                row = ET.SubElement(table, "thead")
            else:
                cell_tag = "td"
                row = ET.SubElement(table, "tr")
        tcell = ET.SubElement(row, cell_tag, attrib=attrib)
        tcell.text = cell["cell text"]

    return str(ET.tostring(table, encoding="unicode", short_empty_elements=False))
