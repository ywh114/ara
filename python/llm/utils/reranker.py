#!/usr/bin/env python3
############################################################################
#                                                                          #
#  Copyright (C) 2025                                                      #
#                                                                          #
#  This program is free software: you can redistribute it and/or modify    #
#  it under the terms of the GNU General Public License as published by    #
#  the Free Software Foundation, either version 3 of the License, or       #
#  (at your option) any later version.                                     #
#                                                                          #
#  This program is distributed in the hope that it will be useful,         #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#  GNU General Public License for more details.                            #
#                                                                          #
#  You should have received a copy of the GNU General Public License       #
#  along with this program. If not, see <http://www.gnu.org/licenses/>.    #
#                                                                          #
############################################################################
# Requires transformers>=4.51.0
# XXX: Demo only.
# See https://huggingface.co/Qwen/Qwen3-Reranker-0.6B.
from abc import ABC, abstractmethod
from typing import Any, override

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.tokenization_utils_base import BatchEncoding
from utils.logger import get_logger

logger = get_logger(__name__)


class RerankerFunction(ABC):
    """Abstract base class for reranker functions."""

    @abstractmethod
    def __init__(
        self,
        model_path: str,
        /,
        model_kwargs: dict[str, str],
        tokenizer_kwargs: dict[str, str],
        **kwargs: Any,
    ) -> None:
        """
        Initialize reranker with model and tokenizer configuration.

        :param model_path: Path to pretrained model.
        :param model_kwargs: Keyword arguments for model loading.
        :param tokenizer_kwargs: Keyword arguments for tokenizer loading.
        """

    @abstractmethod
    def __call__(
        self, instructions: list[tuple[str, str]], /, **kwargs: Any
    ) -> list[float]:
        """
        Reranking operation.

        :return: Reranking results.
        """


class CustomHuggingFaceRerankerFunction(RerankerFunction):
    supported_devices = ('cpu', 'cuda')

    @override
    def __init__(
        self,
        model_path: str,
        /,
        model_kwargs: dict[str, str],
        tokenizer_kwargs: dict[str, str],
        **kwargs: Any,
    ) -> None:
        """
        Initialize HuggingFace reranker with model and tokenizer configuration.

        :param model_path: Path to pretrained model.
        :param model_kwargs: Keyword arguments for model loading.
        :param tokenizer_kwargs: Keyword arguments for tokenizer loading.
        :key device: Computation device ('cpu' or 'cuda').
        :key max_length: Maximum sequence length (default: 8192).
        :raises ValueError: On unsupported device specification.
        """
        self.device = kwargs.get('device', 'cpu')
        if self.device not in self.supported_devices:
            raise ValueError(
                f'Device not supported: {self.device}. '
                f'Supported devices: {self.supported_devices}.'
            )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, **model_kwargs
        )
        if self.device == 'cuda':
            self.model = model.cuda().eval()
        else:
            self.model = model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, **tokenizer_kwargs
        )

        self.max_length = kwargs.get('max_length', 512)

    @override
    def __call__(
        self, instructions: list[tuple[str, str]], /, **kwargs: Any
    ) -> list[float]:
        _ = kwargs
        logger.debug(f'Reranking {len(instructions)} documents.')
        return self.compute_logits(instructions)

    @torch.no_grad
    def compute_logits(self, inputs: list[tuple[str, str]]):
        encoded_inputs: BatchEncoding = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length,
        )
        return (
            self.model(**encoded_inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
            .tolist()
        )


class CustomHuggingFace4Qwen3RerankerFunction(RerankerFunction):
    """
    Demo implementation for `Qwen3-Reranker`.

    Attributes:
        supported_devices: Valid compute devices.
        default_prefix: Default system prompt prefix.
        default_suffix: Default assistant prompt suffix.
        device: Active computation device.
        model: Loaded transformer model.
        tokenizer: Text tokenization component.
        token_false_id: Token ID for 'no' label.
        token_true_id: Token ID for 'yes' label.
        prefix_tokens: Encoded prefix tokens.
        suffix_tokens: Encoded suffix tokens.
        max_length: Maximum sequence length.
        actual_max_length: Effective content length after prompt adjustment.
    """

    supported_devices = ('cpu', 'cuda')
    default_prefix = '<|im_start|>system\nJudge whether the Document meets the '
    'requirements based on the Query and the Instruct provided. Note that the '
    'answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    default_suffix = (
        '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    )

    @override
    def __init__(
        self,
        model_path: str,
        /,
        model_kwargs: dict[str, str],
        tokenizer_kwargs: dict[str, str],
        **kwargs,
    ) -> None:
        """
        Initialize Qwen3 reranker with model and tokenizer configuration.

        :param model_path: Path to pretrained model.
        :param model_kwargs: Keyword arguments for model loading.
        :param tokenizer_kwargs: Keyword arguments for tokenizer loading.
        :key device: Computation device ('cpu' or 'cuda').
        :key prefix: Custom system prompt prefix (d: class default_prefix).
        :key suffix: Custom assistant prompt suffix (d: class default_suffix).
        :key max_length: Maximum sequence length (default: 8192).
        :raises ValueError: On unsupported device specification.
        """
        self.device = kwargs.get('device', 'cpu')
        if self.device not in self.supported_devices:
            raise ValueError(
                f'Device not supported: {self.device}. '
                f'Supported devices: {self.supported_devices}.'
            )

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if self.device == 'cuda':
            self.model = model.cuda().eval()
        else:
            self.model = model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, **tokenizer_kwargs
        )
        self.token_false_id = self.tokenizer.convert_tokens_to_ids('no')
        self.token_true_id = self.tokenizer.convert_tokens_to_ids('yes')
        self.tokenizer.deprecation_warnings[
            'Asking-to-pad-a-fast-tokenizer'
        ] = True  # Disable padding warning for fast tokenizers.

        prefix_text = kwargs.get('prefix', self.default_prefix)
        suffix_text = kwargs.get('prefix', self.default_suffix)
        self.prefix_tokens = self.tokenizer.encode(
            prefix_text, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            suffix_text, add_special_tokens=False
        )

        self.max_length = kwargs.get('max_length', 8192)

        self.actual_max_length = (
            self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )

    def __call__(
        self, instructions: list[tuple[str, str]], /, **kwargs: Any
    ) -> list[float]:
        """
        Rerank documents and compute relevance scores.

        :param instructions: Documents to be reranked.
        :return: List of relevance probabilities [0.0, 1.0].
        """
        _ = kwargs
        logger.debug(f'Reranking {len(instructions)} documents.')
        if instructions:
            return self.compute_logits(
                self.process_inputs(self.reformat_inputs(instructions))
            )
        else:
            return []

    def reformat_inputs(self, instructions: list[tuple[str, str]]) -> list[str]:
        return [f'{task}\n<Document>: {doc}' for task, doc in instructions]

    @torch.no_grad
    def process_inputs(self, reranker_instruct: list[str]) -> BatchEncoding:
        """
        Preprocess documents into model inputs.

        :param reranker_instruct: Documents to process (already wrapped).
        :return: Tokenized inputs.
        """
        inputs = self.tokenizer(
            reranker_instruct,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.actual_max_length,
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = (
                self.prefix_tokens + ele + self.suffix_tokens
            )
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors='pt')
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)

        return inputs

    @torch.no_grad
    def compute_logits(self, inputs: BatchEncoding, **kwargs) -> list[float]:
        """
        Compute relevance probabilities from model outputs.

        :param inputs: Tokenized inputs.
        :return: List of relevance probabilities [0.0, 1.0].
        """
        _ = kwargs
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        return batch_scores[:, 1].exp().tolist()
