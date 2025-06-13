#!/usr/bin/env python3
import os

from llm.utils.openai_api import LLMProfile, LLMWrapper
from llm.utils.stream import CustomStreamHook, CustomStreamHookArgs
from utils.logger import get_logger

from examples.config import confh

logger = get_logger(__name__)


@CustomStreamHook
def print_each_word_hook(csha: CustomStreamHookArgs) -> None:
    head = csha.head_as_str
    print(head, end='', flush=True)


@CustomStreamHook
def contradiction_hook(csha: CustomStreamHookArgs) -> None:
    head = csha.head_as_str
    if 'contradict' in head.lower():
        logger.critical('CONTRADICTION MENTIONED!')


example_profile = LLMProfile(
    os.environ[confh.games.api_example_api_key_env_var],
    confh.games.api_example_api_endpoint,
    model=confh.games.api_example_api_model,
    system_prompt='You are a helpful assistant.',
    stream_hook=print_each_word_hook | contradiction_hook,
)

llm = LLMWrapper()
llm['example'] = example_profile


def example_completion(llm: LLMWrapper) -> None:
    completion = llm.completion(
        'example',
        "Derive Eisenstein's criterion.",
        stream=True,
    )

    text = completion.exhaust()

    logger.debug(f'Reasoning:\n{text.reasoning_content}')
    logger.info(f'Content:\n{text.content}')


if __name__ == '__main__':
    example_completion(llm)
