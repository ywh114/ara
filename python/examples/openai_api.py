#!/usr/bin/env python3
import os

from openai.types.chat import ChatCompletionChunk
from openai.types.shared_params.function_definition import FunctionDefinition

from llm.utils.openai_api import (
    ChatCompletionToolParams,
    LLMProfile,
    LLMWrapper,
)
from llm.utils.stream import (
    CustomFinishHook,
    CustomFinishHookFnType,
    CustomStreamHandler,
    CustomStreamHook,
    CustomHookArgs,
)
from utils.logger import get_logger

from examples.config import confh

logger = get_logger(__name__)


@CustomStreamHook
def print_each_word_hook(ha: CustomHookArgs) -> None:
    head = ha.head_as_str
    print(head, end='', flush=True)


@CustomStreamHook
def contradiction_hook(ha: CustomHookArgs) -> None:
    head = ha.head_as_str
    if 'contradict' in head.lower():
        logger.critical('CONTRADICTION MENTIONED!')


print(CustomFinishHookFnType)


@CustomFinishHook
def weather_provider_hook(ha: CustomHookArgs) -> None:
    pass


example_profile = LLMProfile(
    os.environ[confh.games.api_example_api_key_env_var],
    confh.games.api_example_api_endpoint,
    model=confh.games.api_example_api_model,
    system_prompt='You are a helpful assistant.',
    stream_hook=print_each_word_hook | contradiction_hook,
)

llm = LLMWrapper()
llm['example'] = example_profile


def example_completion(
    llm: LLMWrapper,
) -> CustomStreamHandler[ChatCompletionChunk]:
    get_weather: FunctionDefinition = {
        'name': 'get_weather',
        'description': 'Get weather of an location, the user shoud supply a location first',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The city and state, e.g. San Francisco, CA',
                }
            },
            'required': ['location'],
        },
    }
    get_time: FunctionDefinition = {
        'name': 'get_time',
        'description': 'Get the current time of a location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The city and state, e.g. San Francisco, CA',
                }
            },
            'required': ['location'],
        },
    }
    tools: ChatCompletionToolParams = [
        {
            'type': 'function',
            'function': get_weather,
        },
        {'type': 'function', 'function': get_time},
    ]
    completion = llm.completion(
        'example',
        "What's the weather in Hangzhou?",
        stream=True,
        tools=tools,
    )

    # text = completion.exhaust()

    # logger.debug(f'Reasoning:\n{text.reasoning_content}')
    # logger.info(f'Content:\n{text.content}')

    return completion


if __name__ == '__main__':
    example_completion(llm)
