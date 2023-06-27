import os
from typing import Optional
import numpy as np
from PIL.Image import Image
import torch
from transformers import (
    AutoTokenizer,
    DonutProcessor,
    DonutImageProcessor,
    VisionEncoderDecoderModel,
)
from unstructured_inference.inference.layoutelement import LocationlessLayoutElement
from unstructured_inference.models.unstructuredmodel import UnstructuredElementExtractionModel

MODEL_TYPES = {
    "large_model": {
        "tokenizer_name": "xlm-roberta-large",
        "pre_trained_model_name": "unstructuredio/ved-fine-tuning",
    }
}

LABEL_MAP = [
    "Title",
    "Abstract",
    "Headline",
    "Subheadline",
    "Text",
    "Table",
    "List",
    "List-item",
    "Page number",
    "Header",
    "Footer",
    "Address",
    "Author",
    "Chart",
    "Caption",
    "Formula",
    "Picture",
    "Advertisement",
    "Link",
    "Misc",
    "Field-Name",
    "Value",
    "Threading",
    "Metadata",
    "Form",
]

SPECIAL_TAGS = (
    [f"<s_{label_type}>" for label_type in LABEL_MAP]
    + [f"</s_{label_type}>" for label_type in LABEL_MAP]
    + ["<sep/>"]
)


class UnstructuredLargeModel(UnstructuredElementExtractionModel):
    required_w: int = 1248
    required_h: int = 1664

    def initialize(
        self,
        tokenizer_name: str,
        pre_trained_model_name: str,
        auth_token: Optional[str] = os.environ.get("UNSTRUCTURED_HF_TOKEN"),
    ):
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.processor: DonutProcessor = DonutProcessor(
            image_processor=DonutImageProcessor(
                do_resize=True, size=(self.required_w, self.required_h)
            ),
            tokenizer=self.tokenizer,
        )

        self.tokenizer.add_tokens(new_tokens=SPECIAL_TAGS, special_tokens=True)

        self.model = VisionEncoderDecoderModel.from_pretrained(
            pre_trained_model_name, ignore_mismatched_sizes=True, use_auth_token=auth_token
        )

    def predict(self, image):
        tokens = self.predict_tokens(image)
        elements = self.postprocess(tokens)
        return elements

    def predict_tokens(self, image: Image):
        annotation = self.model.generate(
            self.processor(
                np.array(
                    image,
                    np.float32,
                ),
                return_tensors="pt",
            ).pixel_values,
            decoder_input_ids=torch.tensor([[0]]),
            do_sample=True,
            top_p=0.92,
            top_k=0,
            no_repeat_ngram_size=10,
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
    ):
        elements = []

        # Get special tokens
        tokens_stop = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        tokens_split = self.tokenizer.additional_special_tokens_ids + list(
            self.tokenizer.get_added_vocab().values()
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

        # If exited before bos is achieved
        if start != -1 and start < end:
            slicing_end = end + 1
            string = self.tokenizer.decode(output_ids[start:slicing_end])

            stype = self.tokenizer.decode(last_special_token)

            elements.append(LocationlessLayoutElement(type=stype[3:-1], text=string))

        return elements
