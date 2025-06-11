#!/usr/bin/env python3
import os

from llm.utils.openai_api import APIWrapper
from utils.logger import get_logger

logger = get_logger(__name__)

sk = os.environ['DEEPSEEK_API_KEY']

llm = APIWrapper(
    sk,
    'https://api.deepseek.com',
    model='deepseek-reasoner',
    system_prompt='You are a helpful assistant.',
)

r = llm.completion(
    "Derive Eisenstein's criterion.",
    stream=True,
)


t = True
for _ in r:
    logger.info(r.text)


logger.debug(f'Reasoning:\n{r.text.reasoning_content}')
logger.info(f'Content:\n{r.text.content}')
