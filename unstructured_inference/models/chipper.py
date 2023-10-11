import copy
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL.Image import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers.generation.logits_process import LogitsProcessor

from unstructured_inference.constants import Source
from unstructured_inference.inference.elements import Rectangle
from unstructured_inference.inference.layoutelement import LayoutElement
from unstructured_inference.logger import logger
from unstructured_inference.models.unstructuredmodel import UnstructuredElementExtractionModel
from unstructured_inference.utils import LazyDict

MODEL_TYPES: Dict[Optional[str], Union[LazyDict, dict]] = {
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
        "source": Source.CHIPPER,
    },
}

MODEL_TYPES["chipper"] = MODEL_TYPES["chipperv2"]


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
        self.processor = DonutProcessor.from_pretrained(pre_trained_model_repo, token=auth_token)
        self.tokenizer = self.processor.tokenizer
        self.logits_processor = NoRepeatNGramLogitsProcessor(
            no_repeat_ngram_size,
            get_table_token_ids(self.processor),
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(
            pre_trained_model_repo,
            ignore_mismatched_sizes=True,
            use_auth_token=auth_token,
        )
        if swap_head:
            lm_head_file = hf_hub_download(
                repo_id=pre_trained_model_repo,
                filename="lm_head.pth",
                token=auth_token,
            )
            rank = swap_head_hidden_layer_size
            self.model.decoder.lm_head = torch.nn.Sequential(
                torch.nn.Linear(self.model.decoder.lm_head.weight.shape[1], rank, bias=False),
                torch.nn.Linear(rank, rank, bias=False),
                torch.nn.Linear(rank, self.model.decoder.lm_head.weight.shape[0], bias=True),
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
        elements = self.postprocess(image, tokens, decoder_cross_attentions)
        return elements

    def predict_tokens(
        self,
        image: Image,
    ) -> Tuple[List[int], Sequence[Sequence[torch.Tensor]]]:
        """Predict tokens from image."""
        with torch.no_grad():
            outputs = self.model.generate(
                self.processor(
                    np.array(
                        image,
                        np.float32,
                    ),
                    return_tensors="pt",
                ).pixel_values.to(self.device),
                decoder_input_ids=self.input_ids,
                logits_processor=[self.logits_processor],
                max_length=self.max_length,
                do_sample=True,
                top_p=0.92,
                top_k=5,
                no_repeat_ngram_size=0,
                num_beams=3,
                return_dict_in_generate=True,
                output_attentions=True,
                output_scores=True,
                output_hidden_states=False,
            )

        if "beam_indices" in outputs:
            offset = len(outputs["beam_indices"][0]) - len(outputs["cross_attentions"])

            decoder_cross_attentions = [[torch.Tensor(0)]] * offset

            for token_id in range(0, len(outputs["beam_indices"][0])):
                token_attentions = []

                for decoder_layer_id in range(len(outputs["cross_attentions"][token_id - offset])):
                    token_attentions.append(
                        outputs["cross_attentions"][token_id - offset][decoder_layer_id][
                            outputs["beam_indices"][0][token_id]
                        ].unsqueeze(0),
                    )

                decoder_cross_attentions.append(token_attentions)
        else:
            decoder_cross_attentions = outputs["cross_attentions"]

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
                        tkn_indexes=list(range(start - 1, end)),
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
        if start != -1 and start < end and len(parents) > 0:
            slicing_end = end + 1
            string = self.tokenizer.decode(output_ids[start:slicing_end])

            element = parents.pop(-1)
            element.text = string

            bbox_coords = self.get_bounding_box(
                decoder_cross_attentions=decoder_cross_attentions,
                tkn_indexes=list(range(start - 1, end)),
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

        return elements

    def deduplicate_detected_elements(
        self,
        elements: List[LayoutElement],
        min_text_size: int = 15,
    ) -> List[LayoutElement]:
        """For chipper, remove elements from other sources."""
        return [el for el in elements if el.source in (Source.CHIPPER, Source.CHIPPERV1)]

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
            # shape [4, 1, 16, 1, 1200]->[4, 16, 1200]
            hmaps = hmaps.permute(1, 3, 0, 2, 4).squeeze(0)
            hmaps = hmaps[-1]
            # change shape [4, 16, 1200]->[4, 16, 40, 30] assuming (heatmap_h, heatmap_w) = (40, 30)
            hmaps = hmaps.view(4, 16, heatmap_h, heatmap_w)
            # fusing 16 decoder attention heads i.e. [4, 16, 40, 30]-> [16, 40, 30]
            hmaps = torch.max(hmaps, dim=1)[0]
            # fusing 4 decoder layers from BART i.e. [16, 40, 30]-> [40, 30]
            hmap = torch.max(hmaps, dim=0)[0]

            # dropping discard ratio activations
            flat = hmap.view(heatmap_h * heatmap_w)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), largest=False)
            flat[indices] = 0

            hmap = flat.view(heatmap_h, heatmap_w)

            hmap = hmap.unsqueeze(dim=-1).cpu().numpy()
            hmap = (hmap * 255.0).astype(np.uint8)  # type:ignore
            # (40, 30, 1) uint8
            # fuse heatmaps for different tokens by taking the max
            agg_heatmap = np.max(
                np.asarray([agg_heatmap, cv2.resize(hmap, (final_w, final_h))]),
                axis=0,
            ).astype(np.uint8)

        # threshold to remove small attention pockets
        thres_heatmap = cv2.threshold(agg_heatmap, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        kernel = np.ones((1, 50), np.uint8)
        thres_heatmap = cv2.dilate(thres_heatmap, kernel, iterations=1)
        kernel = np.ones((5, 1), np.uint8)
        thres_heatmap = cv2.erode(thres_heatmap, kernel, iterations=1)

        # Find contours
        contours = cv2.findContours(thres_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        bboxes = [cv2.boundingRect(ctr) for ctr in contours]
        # return box with max area
        x, y, w, h = max(bboxes, key=lambda box: box[2] * box[3])

        return [x, y, x + w, y + h]

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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
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
        banned_tokens = _calc_banned_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
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
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [
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
