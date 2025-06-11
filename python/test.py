#!/usr/bin/env python3
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
model = AutoModelForSequenceClassification.from_pretrained(
    'BAAI/bge-reranker-base'
).eval()

pairs = [
    ['what is panda?', 'hi'],
    [
        'what is panda?',
        'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.',
    ],
    ['what is panda?', 'PANDAS is a python library for scientific computing.'],
    ['what is panda?', 'Pandas are endangered animals.'],
]
with torch.no_grad():
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512,
    )
    scores = (
        model(**inputs, return_dict=True)
        .logits.view(
            -1,
        )
        .float()
        .tolist()
    )
    print(scores)
