import logging
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image as PILImage
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from unstructured_inference.models.unstructuredmodel import UnstructuredModel


class UnstructuredDonutModel(UnstructuredModel):
    """Unstructured model wrapper for Donut image transformer."""

    def predict(self, x: PILImage.Image):
        """Make prediction using donut model"""
        super().predict(x)
        return self.run_prediction(x)

    def initialize(
        self,
        model: Union[str, Path, VisionEncoderDecoderModel] = None,
        processor: Union[str, Path, DonutProcessor] = None,
        config: Optional[Union[str, Path, VisionEncoderDecoderConfig]] = None,
        task_prompt: Optional[str] = "<s>",
        device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Loads the donut model using the specified parameters"""

        self.task_prompt = task_prompt
        self.device = device

        try:
            if not isinstance(config, VisionEncoderDecoderModel):
                config = VisionEncoderDecoderConfig.from_pretrained(config)

            logging.info("Loading the Donut model and processor...")
            self.processor = DonutProcessor.from_pretrained(processor)
            self.model = VisionEncoderDecoderModel.from_pretrained(model, config=config)

        except EnvironmentError:
            logging.critical("Failed to initialize the model.")
            logging.critical(
                "Ensure that the Donut parameters config, model and processor are correct",
            )
            raise ImportError("Review the parameters to initialize a UnstructuredDonutModel obj")
        self.model.to(device)

    def run_prediction(self, x: PILImage.Image):
        """Internal prediction method."""
        pixel_values = self.processor(x, return_tensors="pt").pixel_values
        decoder_input_ids = self.processor.tokenizer(
            self.task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        outputs = self.model.generate(
            pixel_values.to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device),
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        prediction = self.processor.batch_decode(outputs.sequences)[0]
        # NOTE(alan): As of right now I think this would not work if passed in as the model to
        # DocumentLayout.from_file and similar functions that take a model object as input. This
        # produces image-to-text inferences rather than image-to-bboxes, so we actually need to
        # hook it up in a different way.
        prediction = self.processor.token2json(prediction)
        return prediction
