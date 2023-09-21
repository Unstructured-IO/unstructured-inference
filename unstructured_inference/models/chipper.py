from transformers.utils import add_start_docstrings
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
import os
from typing import List, Optional

import numpy as np
import torch
from PIL.Image import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from huggingface_hub import hf_hub_download
from unstructured_inference.inference.layoutelement import LocationlessLayoutElement
from unstructured_inference.models.unstructuredmodel import UnstructuredElementExtractionModel

MODEL_TYPES = {
    "chipper": {
        "pre_trained_model_repo": "unstructuredio/ved-fine-tuning",
        "swap_head": False,
        "prompt": "<s>",
    },
    "chipperv2": {
        "pre_trained_model_repo": "unstructuredio/chipper-fast-fine-tuning",
        "swap_head": True,
        "prompt": "<s><s_hierarchical>",
    },
}


class UnstructuredChipperModel(UnstructuredElementExtractionModel):
    def initialize(
        self,
        pre_trained_model_repo: str,
        swap_head: bool,
        prompt: str,
        no_repeat_ngram_size: int = 10,
        auth_token: Optional[str] = os.environ.get("UNSTRUCTURED_HF_TOKEN"),
    ):
        """Load the model for inference."""
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.processor = DonutProcessor.from_pretrained(pre_trained_model_repo, token=auth_token)
        self.tokenizer = self.processor.tokenizer
        self.logits_processor = NoRepeatNGramLogitsProcessor(
            no_repeat_ngram_size, get_table_token_ids(self.processor)
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(
            pre_trained_model_repo,
            ignore_mismatched_sizes=True,
            use_auth_token=auth_token,
        )
        if swap_head:
            lm_head_file = hf_hub_download(
                repo_id=pre_trained_model_repo, filename="lm_head.pth", token=auth_token
            )
            rank = 128
            self.model.decoder.lm_head = torch.nn.Sequential(
                torch.nn.Linear(self.model.decoder.lm_head.weight.shape[1], rank, bias=False),
                torch.nn.Linear(rank, rank, bias=False),
                torch.nn.Linear(rank, self.model.decoder.lm_head.weight.shape[0], bias=True),
            )
            self.model.decoder.lm_head.load_state_dict(torch.load(lm_head_file))

        self.input_ids = self.processor.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image) -> List[LocationlessLayoutElement]:
        """Do inference using the wrapped model."""
        tokens = self.predict_tokens(image)
        elements = self.postprocess(tokens)
        return elements

    def predict_tokens(self, image: Image) -> List[int]:
        """Predict tokens from image."""
        with torch.no_grad():
            annotation = self.model.generate(
                self.processor(
                    np.array(
                        image,
                        np.float32,
                    ),
                    return_tensors="pt",
                ).pixel_values.to(self.device),
                decoder_input_ids=self.input_ids,
                logits_processor=[self.logits_processor],
                do_sample=True,
                top_p=0.92,
                top_k=5,
                no_repeat_ngram_size=0,
                num_beams=3,
            ).tolist()[0]

        tokens = (
            [self.processor.tokenizer.bos_token_id]
            + [
                e
                for e in annotation
                if e != self.processor.tokenizer.bos_token_id
                and e != self.processor.tokenizer.eos_token_id
            ]
            + [self.processor.tokenizer.eos_token_id]
        )

        return tokens

    def postprocess(
        self,
        output_ids,
    ) -> List[LocationlessLayoutElement]:
        """Process tokens into layout elements."""
        elements = []

        # Get special tokens
        tokens_stop = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        tokens_split = self.tokenizer.additional_special_tokens_ids + list(
            self.tokenizer.get_added_vocab().values(),
        )

        start = end = -1
        last_special_token = None

        # Get bboxes - skip first token - bos
        for i in range(1, len(output_ids)):
            # Finish bounding box generation
            if output_ids[i] in tokens_stop:
                break
            if output_ids[i] in tokens_split:
                if start != -1 and start < end:
                    slicing_end = end + 1
                    string = self.tokenizer.decode(output_ids[start:slicing_end])

                    stype = self.tokenizer.decode(last_special_token)

                    elements.append(LocationlessLayoutElement(type=stype[3:-1], text=string))

                start = -1
                last_special_token = output_ids[i]
            else:
                if start == -1:
                    start = i

                end = i

        # If exited before eos is achieved
        if start != -1 and start < end:
            slicing_end = end + 1
            string = self.tokenizer.decode(output_ids[start:slicing_end])

            stype = self.tokenizer.decode(last_special_token)

            elements.append(LocationlessLayoutElement(type=stype[3:-1], text=string))

        return elements


# Inspired on https://github.com/huggingface/transformers/blob/8e3980a290acc6d2f8ea76dba111b9ef0ef00309/src/transformers/generation/logits_process.py#L706
class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, ngram_size: int, skip_tokens=None):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
            )
        self.ngram_size = ngram_size
        self.skip_tokens = skip_tokens

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
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
    input_ids, cur_len, logits, batch_size=1, no_repeat_ngram_size=0, skip_tokens=None
):
    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
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


def _calc_banned_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
    # Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [
                ngram[-1]
            ]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())

        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def get_table_token_ids(processor):
    skip_tokens = {
        token_id
        for token, token_id in processor.tokenizer.get_added_vocab().items()
        if token.startswith("<t")
    }
    return skip_tokens
