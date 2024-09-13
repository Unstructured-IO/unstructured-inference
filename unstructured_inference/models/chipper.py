import copy
import os
import platform
from contextlib import nullcontext
from typing import ContextManager, Dict, List, Optional, Sequence, TextIO, Tuple, Union

import cv2
import numpy as np
import torch
import transformers
from cv2.typing import MatLike
from PIL.Image import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteria

from unstructured_inference.constants import CHIPPER_VERSIONS, Source
from unstructured_inference.inference.elements import Rectangle
from unstructured_inference.inference.layoutelement import LayoutElement
from unstructured_inference.logger import logger
from unstructured_inference.models.unstructuredmodel import (
    UnstructuredElementExtractionModel,
)
from unstructured_inference.utils import LazyDict, download_if_needed_and_get_local_path, strip_tags

MODEL_TYPES: Dict[str, Union[LazyDict, dict]] = {
    "chipperv1": {
        "pre_trained_model_repo": "unstructuredio/ved-fine-tuning",
        "swap_head": False,
        "start_token_prefix": "<s_",
        "prompt": "<s>",
        "max_length": 1200,
        "heatmap_h": 52,
        "heatmap_w": 39,
        "source": Source.CHIPPERV1,
    },
    "chipperv2": {
        "pre_trained_model_repo": "unstructuredio/chipper-fast-fine-tuning",
        "swap_head": True,
        "swap_head_hidden_layer_size": 128,
        "start_token_prefix": "<s_",
        "prompt": "<s><s_hierarchical>",
        "max_length": 1536,
        "heatmap_h": 40,
        "heatmap_w": 30,
        "source": Source.CHIPPERV2,
    },
    "chipperv3": {
        "pre_trained_model_repo": "unstructuredio/chipper-v3",
        "swap_head": True,
        "swap_head_hidden_layer_size": 128,
        "start_token_prefix": "<s_",
        "prompt": "<s><s_hierarchical>",
        "max_length": 1536,
        "heatmap_h": 40,
        "heatmap_w": 30,
        "source": Source.CHIPPER,
    },
}

MODEL_TYPES["chipper"] = MODEL_TYPES["chipperv3"]


class UnstructuredChipperModel(UnstructuredElementExtractionModel):
    def initialize(
        self,
        pre_trained_model_repo: str,
        start_token_prefix: str,
        prompt: str,
        max_length: int,
        heatmap_h: int,
        heatmap_w: int,
        source: Source,
        swap_head: bool = False,
        swap_head_hidden_layer_size: int = 0,
        no_repeat_ngram_size: int = 10,
        auth_token: Optional[str] = os.environ.get("UNSTRUCTURED_HF_TOKEN"),
        device: Optional[str] = None,
    ):
        """Load the model for inference."""
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.max_length = max_length
        self.heatmap_h = heatmap_h
        self.heatmap_w = heatmap_w
        self.source = source
        self.processor = DonutProcessor.from_pretrained(
            pre_trained_model_repo,
            token=auth_token,
        )
        self.tokenizer = self.processor.tokenizer
        self.logits_processor = [
            NoRepeatNGramLogitsProcessor(
                no_repeat_ngram_size,
                get_table_token_ids(self.processor),
            ),
        ]

        self.stopping_criteria = [
            NGramRepetitonStoppingCriteria(
                repetition_window=30,
                skip_tokens=get_table_token_ids(self.processor),
            ),
        ]

        self.model = VisionEncoderDecoderModel.from_pretrained(
            pre_trained_model_repo,
            ignore_mismatched_sizes=True,
            token=auth_token,
        )
        if swap_head:
            lm_head_file = download_if_needed_and_get_local_path(
                path_or_repo=pre_trained_model_repo,
                filename="lm_head.pth",
                token=auth_token,
            )
            rank = swap_head_hidden_layer_size
            self.model.decoder.lm_head = torch.nn.Sequential(
                torch.nn.Linear(
                    self.model.decoder.lm_head.weight.shape[1],
                    rank,
                    bias=False,
                ),
                torch.nn.Linear(rank, rank, bias=False),
                torch.nn.Linear(
                    rank,
                    self.model.decoder.lm_head.weight.shape[0],
                    bias=True,
                ),
            )
            self.model.decoder.lm_head.load_state_dict(torch.load(lm_head_file))
        else:
            if swap_head_hidden_layer_size is not None:
                logger.warning(
                    f"swap_head is False but recieved value {swap_head_hidden_layer_size} for "
                    "swap_head_hidden_layer_size, which will be ignored.",
                )

        self.input_ids = self.processor.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

        self.start_tokens = [
            v
            for k, v in self.processor.tokenizer.get_added_vocab().items()
            if k.startswith(start_token_prefix) and v not in self.input_ids
        ]
        self.end_tokens = [
            v for k, v in self.processor.tokenizer.get_added_vocab().items() if k.startswith("</s_")
        ]
        self.tokens_stop = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]

        self.model.to(self.device)

        self.model.eval()

    def predict(self, image) -> List[LayoutElement]:
        """Do inference using the wrapped model."""
        tokens, decoder_cross_attentions = self.predict_tokens(image)
        elements = self.format_table_elements(
            self.postprocess(image, tokens, decoder_cross_attentions),
        )
        return elements

    @staticmethod
    def format_table_elements(elements: List[LayoutElement]) -> List[LayoutElement]:
        """Makes chipper table element return the same as other layout models.

        1. If `text` attribute is an html (has html tags in it), copies the `text`
        attribute to `text_as_html` attribute.
        2. Strips html tags from the `text` attribute.
        """
        for element in elements:
            text = strip_tags(element.text) if element.text is not None else element.text
            if text != element.text:
                element.text_as_html = element.text  # type: ignore[attr-defined]
            element.text = text
        return elements

    def predict_tokens(
        self,
        image: Image,
    ) -> Tuple[List[int], Sequence[Sequence[torch.Tensor]]]:
        """Predict tokens from image."""
        transformers.set_seed(42)

        with torch.no_grad():
            amp: Union[TextIO, ContextManager[None]] = (
                torch.cuda.amp.autocast()
                if self.device == "cuda"
                else (torch.cpu.amp.autocast() if platform.machine() == "x86_64" else nullcontext())
            )
            with amp:
                encoder_outputs = self.model.encoder(
                    self.processor(
                        image,
                        return_tensors="pt",
                    ).pixel_values.to(self.device),
                )

                outputs = self.model.generate(
                    encoder_outputs=encoder_outputs,
                    input_ids=self.input_ids,
                    max_length=self.max_length,
                    no_repeat_ngram_size=0,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_attentions=True,
                    output_scores=True,
                    output_hidden_states=False,
                    stopping_criteria=self.stopping_criteria,
                )

                if (
                    len(outputs["sequences"][0]) < self.max_length
                    and outputs["sequences"][0][-1] != self.processor.tokenizer.eos_token_id
                ):
                    outputs = self.model.generate(
                        encoder_outputs=encoder_outputs,
                        input_ids=self.input_ids,
                        max_length=self.max_length,
                        logits_processor=self.logits_processor,
                        do_sample=True,
                        no_repeat_ngram_size=0,
                        num_beams=3,
                        return_dict_in_generate=True,
                        output_attentions=True,
                        output_scores=True,
                        output_hidden_states=False,
                    )

        offset = len(self.input_ids)

        if "beam_indices" in outputs:
            decoder_cross_attentions = [[torch.Tensor(0)]] * offset

            for token_id in range(0, len(outputs["beam_indices"][0])):
                if outputs["beam_indices"][0][token_id] == -1:
                    break

                token_attentions = []

                for decoder_layer_id in range(
                    len(outputs["cross_attentions"][token_id]),
                ):
                    token_attentions.append(
                        outputs["cross_attentions"][token_id][decoder_layer_id][
                            outputs["beam_indices"][0][token_id]
                        ].unsqueeze(0),
                    )

                decoder_cross_attentions.append(token_attentions)
        else:
            decoder_cross_attentions = [[torch.Tensor(0)]] * offset + list(
                outputs["cross_attentions"],
            )

        tokens = outputs["sequences"][0]

        return tokens, decoder_cross_attentions

    def update_parent_bbox(self, element):
        """Update parents bboxes in a recursive way"""
        # Check if children, if so update parent bbox
        if element.parent is not None:
            parent = element.parent
            # parent has no box, create one
            if parent.bbox is None:
                parent.bbox = copy.copy(element.bbox)
            else:
                # adjust parent box
                parent.bbox.x1 = min(parent.bbox.x1, element.bbox.x1)
                parent.bbox.y1 = min(parent.bbox.y1, element.bbox.y1)
                parent.bbox.x2 = max(parent.bbox.x2, element.bbox.x2)
                parent.bbox.y2 = max(parent.bbox.y2, element.bbox.y2)

            self.update_parent_bbox(element.parent)

    def postprocess(
        self,
        image: Image,
        output_ids: List[int],
        decoder_cross_attentions: Sequence[Sequence[torch.Tensor]],
    ) -> List[LayoutElement]:
        """Process tokens into layout elements."""
        elements: List[LayoutElement] = []
        parents: List[LayoutElement] = []
        start = end = -1

        x_offset, y_offset, ratio = self.image_padding(
            image.size,
            (
                self.processor.image_processor.size["width"],
                self.processor.image_processor.size["height"],
            ),
        )

        # Get bboxes - skip first token - bos
        for i in range(1, len(output_ids)):
            # Finish bounding box generation
            if output_ids[i] in self.tokens_stop:
                break
            if output_ids[i] in self.start_tokens:
                # Create the element
                stype = self.tokenizer.decode(output_ids[i])
                # Identify parent
                parent = parents[-1] if len(parents) > 0 else None
                # Add to parent list
                element = LayoutElement(
                    parent=parent,
                    type=stype[3:-1],
                    text="",
                    bbox=None,  # type: ignore
                    source=self.source,
                )

                parents.append(element)
                elements.append(element)
                start = -1
            elif output_ids[i] in self.end_tokens:
                if start != -1 and start <= end and len(parents) > 0:
                    slicing_end = end + 1
                    string = self.tokenizer.decode(output_ids[start:slicing_end])

                    element = parents.pop(-1)

                    element.text = string
                    bbox_coords = self.get_bounding_box(
                        decoder_cross_attentions=decoder_cross_attentions,
                        tkn_indexes=list(range(start, end + 1)),
                        final_w=self.processor.image_processor.size["width"],
                        final_h=self.processor.image_processor.size["height"],
                        heatmap_w=self.heatmap_w,
                        heatmap_h=self.heatmap_h,
                    )

                    bbox_coords = self.adjust_bbox(
                        bbox_coords,
                        x_offset,
                        y_offset,
                        ratio,
                        image.size,
                    )

                    element.bbox = Rectangle(*bbox_coords)

                    self.update_parent_bbox(element)

                start = -1
            else:
                if start == -1:
                    start = i

                end = i

        # If exited before eos is achieved
        if start != -1 and start <= end and len(parents) > 0:
            slicing_end = end + 1
            string = self.tokenizer.decode(output_ids[start:slicing_end])

            element = parents.pop(-1)
            element.text = string

            bbox_coords = self.get_bounding_box(
                decoder_cross_attentions=decoder_cross_attentions,
                tkn_indexes=list(range(start, end + 1)),
                final_w=self.processor.image_processor.size["width"],
                final_h=self.processor.image_processor.size["height"],
                heatmap_w=self.heatmap_w,
                heatmap_h=self.heatmap_h,
            )

            bbox_coords = self.adjust_bbox(
                bbox_coords,
                x_offset,
                y_offset,
                ratio,
                image.size,
            )

            element.bbox = Rectangle(*bbox_coords)

            self.update_parent_bbox(element)

        # Reduce bounding boxes
        for element in elements:
            self.reduce_element_bbox(image, elements, element)

        # Solve overlaps
        self.resolve_bbox_overlaps(image, elements)

        return elements

    def deduplicate_detected_elements(
        self,
        elements: List[LayoutElement],
        min_text_size: int = 15,
    ) -> List[LayoutElement]:
        """For chipper, remove elements from other sources."""
        return [el for el in elements if el.source in CHIPPER_VERSIONS]

    def adjust_bbox(self, bbox, x_offset, y_offset, ratio, target_size):
        """Translate bbox by (x_offset, y_offset) and shrink by ratio."""
        return [
            max((bbox[0] - x_offset) / ratio, 0),
            max((bbox[1] - y_offset) / ratio, 0),
            min((bbox[2] - x_offset) / ratio, target_size[0]),
            min((bbox[3] - y_offset) / ratio, target_size[1]),
        ]

    def get_bounding_box(
        self,
        decoder_cross_attentions: Sequence[Sequence[torch.Tensor]],
        tkn_indexes: List[int],
        final_h: int = 1280,
        final_w: int = 960,
        heatmap_h: int = 40,
        heatmap_w: int = 30,
        discard_ratio: float = 0.99,
    ) -> List[int]:
        """
        decoder_cross_attention: tuple(tuple(torch.FloatTensor))
        Tuple (one element for each generated token) of tuples (one element for
        each layer of the decoder) of  `torch.FloatTensor` of shape
        `(batch_size, num_heads, generated_length, sequence_length)`
        """
        agg_heatmap = np.zeros([final_h, final_w], dtype=np.uint8)

        for tidx in tkn_indexes:
            hmaps = torch.stack(list(decoder_cross_attentions[tidx]), dim=0)
            # shape [4, 1, 16, 1, 1200] -> [1, 4, 16, 1200]
            hmaps = hmaps.permute(1, 3, 0, 2, 4).squeeze(0)
            # shape [4, 1, 16, 1, 1200]->[4, 16, 1200]
            hmaps = hmaps[-1]
            # change shape [4, 16, 1200]->[4, 16, 40, 30] assuming (heatmap_h, heatmap_w) = (40, 30)
            hmaps = hmaps.view(4, 16, heatmap_h, heatmap_w)
            hmaps = torch.where(hmaps > 0.65, hmaps, 0.0)
            # fusing 16 decoder attention heads i.e. [4, 16, 40, 30]-> [4, 40, 30]
            hmaps = torch.max(hmaps, dim=1)[0]
            # hmaps = torch.mean(hmaps, dim=1)
            # fusing 4 decoder layers from BART i.e. [4, 40, 30]-> [4, 40, 30]
            hmap = torch.max(hmaps, dim=0)[0]
            # hmap = torch.mean(hmaps, dim=0)

            # dropping discard ratio activations
            flat = hmap.view(heatmap_h * heatmap_w)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), largest=False)
            flat[indices] = 0

            hmap = flat.view(heatmap_h, heatmap_w)

            hmap = hmap.unsqueeze(dim=-1).to(torch.float32).cpu().numpy()  # type: ignore
            hmap = (hmap * 255.0).astype(np.uint8)  # type:ignore
            # (40, 30, 1) uint8
            # fuse heatmaps for different tokens by taking the max
            agg_heatmap = np.max(
                np.asarray(
                    [
                        agg_heatmap,
                        cv2.resize(  # type: ignore
                            hmap,
                            (final_w, final_h),
                            interpolation=cv2.INTER_LINEAR_EXACT,  # cv2.INTER_CUBIC
                        ),
                    ],
                ),
                axis=0,
            ).astype(np.uint8)

        # threshold to remove small attention pockets
        thres_heatmap = cv2.threshold(
            agg_heatmap,
            200,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

        # kernel = np.ones((1, 20), np.uint8)
        # thres_heatmap = cv2.dilate(thres_heatmap, kernel, iterations=2)

        """
        kernel = np.ones((5, 1), np.uint8)
        thres_heatmap = cv2.erode(thres_heatmap, kernel, iterations=1)
        """
        # Find contours
        contours = cv2.findContours(
            thres_heatmap,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours_selection = contours[0] if len(contours) == 2 else contours[1]

        bboxes = [cv2.boundingRect(ctr) for ctr in contours_selection]

        if len(bboxes) > 1:
            kernel = np.ones((1, 50), np.uint8)
            thres_heatmap = cv2.dilate(thres_heatmap, kernel, iterations=1)

            contours = cv2.findContours(
                thres_heatmap,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contours_selection = contours[0] if len(contours) == 2 else contours[1]

            bboxes = [cv2.boundingRect(ctr) for ctr in contours_selection]

        try:
            # return box with max area
            x, y, w, h = max(bboxes, key=lambda box: box[2] * box[3])

            return [x, y, x + w, y + h]
        except ValueError:
            return [0, 0, 1, 1]

    def reduce_element_bbox(
        self,
        image: Image,
        elements: List[LayoutElement],
        element: LayoutElement,
    ):
        """
        Given a LayoutElement element, reduce the size of the bounding box,
        depending on existing elements
        """
        if element.bbox:
            bbox = [element.bbox.x1, element.bbox.y1, element.bbox.x2, element.bbox.y2]

            if not self.element_overlap(elements, element):
                element.bbox = Rectangle(*self.reduce_bbox_no_overlap(image, bbox))
            else:
                element.bbox = Rectangle(*self.reduce_bbox_overlap(image, bbox))

    def bbox_overlap(
        self,
        bboxa: List[float],
        bboxb: List[float],
    ) -> bool:
        """
        Check if bounding boxes bboxa and bboxb overlap
        """
        x1_a, y1_a, x2_a, y2_a = bboxa
        x1_b, y1_b, x2_b, y2_b = bboxb

        return bool(x1_a <= x2_b and x1_b <= x2_a and y1_a <= y2_b and y1_b <= y2_a)

    def element_overlap(
        self,
        elements: List[LayoutElement],
        element: LayoutElement,
    ) -> bool:
        """
        Check if an element overlaps with existing elements
        """
        bboxb = [
            element.bbox.x1,
            element.bbox.y1,
            element.bbox.x2,
            max(element.bbox.y1, element.bbox.y2),
        ]

        for check_element in elements:
            if check_element == element:
                continue

            if self.bbox_overlap(
                [
                    check_element.bbox.x1,
                    check_element.bbox.y1,
                    check_element.bbox.x2,
                    max(check_element.bbox.y1, check_element.bbox.y2),
                ],
                bboxb,
            ):
                return True

        return False

    def remove_horizontal_lines(
        self,
        img_dst: MatLike,
    ) -> MatLike:
        """
        Remove horizontal lines in an image
        """
        gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_dst, 50, 150, apertureSize=5)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))

        # Using morph open to get lines inside the drawing
        opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_selection = cnts[0] if len(cnts) == 2 else cnts[1]
        mask = np.zeros(gray_dst.shape, np.uint8)
        for c in cnts_selection:
            cv2.drawContours(mask, [c], -1, (255, 255, 255), 6)

        return cv2.inpaint(img_dst, mask, 3, cv2.INPAINT_TELEA)

    def reduce_bbox_no_overlap(
        self,
        image: Image,
        input_bbox: List[float],
    ) -> List[float]:
        """
        If an element does not overlap with other elements, remove any empty space around it
        """
        input_bbox = [int(b) for b in input_bbox]

        if (
            (input_bbox[2] * input_bbox[3] <= 0)
            or (input_bbox[2] < input_bbox[0])
            or (input_bbox[3] < input_bbox[1])
        ):
            return input_bbox

        nimage = np.array(image.crop(input_bbox))  # type: ignore

        nimage = self.remove_horizontal_lines(nimage)

        # Convert the image to grayscale
        gray = cv2.bitwise_not(cv2.cvtColor(nimage, cv2.COLOR_BGR2GRAY))

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find the contours in the edge image
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        try:
            largest_contour = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(largest_contour)

        except ValueError:
            return input_bbox

        return [
            input_bbox[0] + x,
            input_bbox[1] + y,
            input_bbox[0] + x + w,
            input_bbox[1] + y + h,
        ]

    def reduce_bbox_overlap(
        self,
        image: Image,
        input_bbox: List[float],
    ) -> List[float]:
        """
        If an element does overlap with other elements, reduce bouding box by selecting the largest
        bbox after blurring existing text
        """
        input_bbox = [int(b) for b in input_bbox]

        if (
            (input_bbox[2] * input_bbox[3] <= 0)
            or (input_bbox[2] < input_bbox[0])
            or (input_bbox[3] < input_bbox[1])
        ):
            return input_bbox

        nimage = np.array(image.crop(input_bbox))  # type: ignore

        nimage = self.remove_horizontal_lines(nimage)

        center_h = nimage.shape[0] / 2
        center_w = nimage.shape[1] / 2

        gray = cv2.bitwise_not(cv2.cvtColor(nimage, cv2.COLOR_BGR2GRAY))
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        nim = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)[1]
        binary_mask = cv2.threshold(nim, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        if center_h > center_w:
            kernel = np.ones((1, 80), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            kernel = np.ones((40, 1), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        else:
            # kernel = np.ones((8, 1), np.uint8)
            # binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

            kernel = np.ones((1, 80), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            kernel = np.ones((30, 1), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        contours = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        contours_selection = contours[0] if len(contours) == 2 else contours[1]

        bboxes = [cv2.boundingRect(ctr) for ctr in contours_selection]

        nbboxes = [
            bbox
            for bbox in bboxes
            if bbox[2] * bbox[3] > 1
            and (bbox[0] < center_w and bbox[0] + bbox[2] > center_w)
            and (bbox[1] < center_h and bbox[1] + bbox[3] > center_h)
        ]

        if len(nbboxes) == 0:
            nbboxes = [bbox for bbox in bboxes if bbox[2] * bbox[3] > 1]

        if len(nbboxes) == 0:
            return input_bbox

        x, y, w, h = max(nbboxes, key=lambda box: box[2] * box[3])

        return [
            input_bbox[0] + x,
            input_bbox[1] + y,
            input_bbox[0] + x + w,
            input_bbox[1] + y + h,
        ]

    def separation_margins(
        self,
        mapping: MatLike,
    ) -> Optional[List[Tuple[int, int, int]]]:
        """
        Find intervals with no content
        """
        current_start = None

        margins = []

        for i, value in enumerate(mapping):
            if value[0] == 0:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    if current_start > 0:
                        margins.append((current_start, i, i - current_start))
                    current_start = None

        if current_start is not None:
            margins.append((current_start, i, i - current_start))

        return margins

    def largest_margin(
        self,
        image: Image,
        input_bbox: Tuple[float, float, float, float],
        transpose: bool = False,
    ) -> Optional[Tuple[int, int, int]]:
        """
        Find the largest region with no text
        """
        if (
            (input_bbox[2] * input_bbox[3] <= 0)
            or (input_bbox[2] < input_bbox[0])
            or (input_bbox[3] < input_bbox[1])
        ):
            return None

        nimage = np.array(image.crop(input_bbox))  # type: ignore

        if nimage.shape[0] * nimage.shape[1] == 0:
            return None

        if transpose:
            nimage = np.swapaxes(nimage, 0, 1)

        nimage = self.remove_horizontal_lines(nimage)

        gray = cv2.bitwise_not(cv2.cvtColor(nimage, cv2.COLOR_BGR2GRAY))

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        mapping = cv2.reduce(edges, 1, cv2.REDUCE_MAX)

        margins = self.separation_margins(mapping)

        if margins is None or len(margins) == 0:
            return None

        largest_margin = sorted(margins, key=lambda x: x[2], reverse=True)[0]

        return largest_margin

    def check_overlap(
        self,
        box1: List[float],
        box2: List[float],
    ) -> Tuple[
        str,
        List[float],
        List[float],
        List[float],
        List[float],
        Optional[Tuple[float, float, float, float]],
    ]:
        """
        Check the overlap between two bounding boxes, return properties of the overlap and
        the bbox of the overlapped region
        """
        # Get the coordinates of the two boxes
        x1, y1, x11, y11 = box1
        x2, y2, x21, y21 = box2

        w1 = x11 - x1
        h1 = y11 - y1
        w2 = x21 - x2
        h2 = y21 - y2

        # Check for horizontal overlap
        horizontal_overlap = bool(x1 <= x2 + w2 and x1 + w1 >= x2)

        # Check for vertical overlap
        vertical_overlap = bool(y1 <= y2 + h2 and y1 + h1 >= y2)

        # Check for both horizontal and vertical overlap
        if horizontal_overlap and vertical_overlap:
            overlap_type = "both"
            intersection_x1 = max(x1, x2)
            intersection_y1 = max(y1, y2)
            intersection_x2 = min(x1 + w1, x2 + w2)
            intersection_y2 = min(y1 + h1, y2 + h2)
            overlapping_bbox = (
                intersection_x1,
                intersection_y1,
                intersection_x2,
                intersection_y2,
            )
        elif horizontal_overlap and not vertical_overlap:
            overlap_type = "horizontal"
            overlapping_bbox = None
        elif not horizontal_overlap and vertical_overlap:
            overlap_type = "vertical"
            overlapping_bbox = None
        else:
            overlap_type = "none"
            overlapping_bbox = None

        # Check which box is on top and/or left
        if y1 < y2:
            top_box = box1
            bottom_box = box2
        else:
            top_box = box2
            bottom_box = box1

        if x1 < x2:
            left_box = box1
            right_box = box2
        else:
            left_box = box2
            right_box = box1

        return overlap_type, top_box, bottom_box, left_box, right_box, overlapping_bbox

    def resolve_bbox_overlaps(
        self,
        image: Image,
        elements: List[LayoutElement],
    ):
        """
        Resolve overlapping bounding boxes
        """
        for element in elements:
            if element.parent is not None:
                continue

            ebbox1 = element.bbox
            if ebbox1 is None:
                continue
            bbox1 = [ebbox1.x1, ebbox1.y1, ebbox1.x2, max(ebbox1.y1, ebbox1.y2)]

            for celement in elements:
                if element == celement or celement.parent is not None:
                    continue

                ebbox2 = celement.bbox
                bbox2 = [ebbox2.x1, ebbox2.y1, ebbox2.x2, max(ebbox2.y1, ebbox2.y2)]

                if self.bbox_overlap(bbox1, bbox2):
                    check = self.check_overlap(bbox1, bbox2)

                    # For resolution, we should be sure that the overlap in the other dimension
                    # is large
                    if (
                        check[-1]
                        and check[-1][0] > 0
                        # and (check[0] == "vertical" or check[0] == "both")
                        and (bbox1[2] - bbox1[0]) / check[-1][0] > 0.9
                        and (bbox2[2] - bbox2[0]) / check[-1][0] > 0.9
                    ):
                        margin = self.largest_margin(image, check[-1])

                        if margin:
                            # Check with box is on top
                            if bbox1 == check[1]:
                                bbox1[3] -= margin[0]
                                bbox2[1] += margin[1]
                            else:
                                bbox2[3] -= margin[0]
                                bbox1[1] += margin[1]

                            element.bbox = Rectangle(*bbox1)
                            celement.bbox = Rectangle(*bbox2)

                        check = self.check_overlap(bbox1, bbox2)

                        # We need to identify cases with horizontal alignment after
                        # vertical resolution. This is commented out for now.
                        """
                        # For resolution, we should be sure that the overlap in the other dimension
                        # is large
                        if (
                            check[-1]
                            and check[-1][0] > 0
                            and (bbox1[3] - bbox1[1]) / check[-1][1] > 0.9
                            and (bbox2[3] - bbox2[1]) / check[-1][1] > 0.9
                        ):
                            margin = self.largest_margin(
                                image,
                                check[-1],
                                transpose=True,
                            )
                            if margin:
                                # Check with box is on top
                                if bbox1 == check[3]:
                                    bbox1[2] -= margin[0]
                                    bbox2[0] += margin[1]
                                else:
                                    bbox2[2] -= margin[0]
                                    bbox1[0] += margin[1]

                                element.bbox = Rectangle(*bbox1)
                                celement.bbox = Rectangle(*bbox2)
                        """

    def image_padding(self, input_size, target_size):
        """
        Resize an image to a defined size, preserving the aspect ratio,
        and pad with a background color to maintain aspect ratio.

        Args:
            input_size (tuple): Size of the input in the format (width, height).
            target_size (tuple): Desired target size in the format (width, height).
        """
        # Calculate the aspect ratio of the input and target sizes
        input_width, input_height = input_size  # 612, 792
        target_width, target_height = target_size  # 960, 1280
        aspect_ratio_input = input_width / input_height  # 0.773
        aspect_ratio_target = target_width / target_height  # 0.75

        # Determine the size of the image when resized to fit within the
        # target size while preserving the aspect ratio
        if aspect_ratio_input > aspect_ratio_target:
            # Resize the image to fit the target width and calculate the new height
            new_width = target_width
            new_height = int(new_width / aspect_ratio_input)
        else:
            # Resize the image to fit the target height and calculate the new width
            new_height = target_height
            new_width = int(new_height * aspect_ratio_input)

        # Calculate the position to paste the resized image to center it within the target size
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        return x_offset, y_offset, new_width / input_width


# Inspired on
# https://github.com/huggingface/transformers/blob/8e3980a290acc6d2f8ea76dba111b9ef0ef00309/src/transformers/generation/logits_process.py#L706
class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, ngram_size: int, skip_tokens: Optional[Sequence[int]] = None):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}",
            )
        self.ngram_size = ngram_size
        self.skip_tokens = skip_tokens

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits
                for each vocabulary when not using beam search or log softmax for
                each vocabulary token when using beam search

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The
            processed prediction scores.

        """
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        return _no_repeat_ngram_logits(
            input_ids,
            cur_len,
            scores,
            batch_size=num_batch_hypotheses,
            no_repeat_ngram_size=self.ngram_size,
            skip_tokens=self.skip_tokens,
        )


class NGramRepetitonStoppingCriteria(StoppingCriteria):
    def __init__(self, repetition_window: int, skip_tokens: set = set()):
        self.repetition_window = repetition_window
        self.skip_tokens = skip_tokens

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`]
                and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be scores for each
                vocabulary token before SoftMax or scores for each vocabulary token after SoftMax.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional stopping criteria specific kwargs.

        Return:
            `bool`. `False` indicates we should continue, `True` indicates we should stop.

        """
        num_batch_hypotheses = input_ids.shape[0]
        cur_len = input_ids.shape[-1]

        for banned_tokens in _calc_banned_tokens(
            input_ids,
            num_batch_hypotheses,
            self.repetition_window,
            cur_len,
        ):
            for token in banned_tokens:
                if token not in self.skip_tokens:
                    return True

        return False


def _no_repeat_ngram_logits(
    input_ids: torch.LongTensor,
    cur_len: int,
    logits: torch.FloatTensor,
    batch_size: int = 1,
    no_repeat_ngram_size: int = 0,
    skip_tokens: Optional[Sequence[int]] = None,
) -> torch.FloatTensor:
    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        # from fairseq:
        # https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_tokens = _calc_banned_tokens(
            input_ids,
            batch_size,
            no_repeat_ngram_size,
            cur_len,
        )
        for batch_idx in range(batch_size):
            if skip_tokens is not None:
                logits[
                    batch_idx,
                    [token for token in banned_tokens[batch_idx] if int(token) not in skip_tokens],
                ] = -float("inf")
            else:
                logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

    return logits


def _calc_banned_tokens(
    prev_input_ids: torch.LongTensor,
    num_hypos: int,
    no_repeat_ngram_size: int,
    cur_len: int,
) -> List[Tuple[int, ...]]:
    # Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [() for _ in range(num_hypos)]
    generated_ngrams: List[Dict[Tuple[int, ...], List[int]]] = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                prev_ngram_tuple,
                [],
            ) + [
                ngram[-1],
            ]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())

        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def get_table_token_ids(processor):
    """
    Extracts the table tokens from the tokenizer of the processor

    Args:
        processor (DonutProcessor): processor used to pre-process the images and text.
    """
    skip_tokens = {
        token_id
        for token, token_id in processor.tokenizer.get_added_vocab().items()
        if token.startswith("<t") or token.startswith("</t")
    }
    return skip_tokens
