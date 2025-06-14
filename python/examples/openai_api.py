#!/usr/bin/env python3
from datetime import datetime
import os
from typing import Any

from llm.utils.context_manager import MultiroundContextManager
from llm.utils.openai_api import (
    ChatCompletionToolParams,
    LLMProfile,
    LLMWrapper,
    find_last_tool,
)
from llm.utils.stream import (
    CustomCaptureFinish,
    CustomHookArgs,
    CustomStreamHandler,
    CustomStreamHook,
    CustomToolHook,
    ToolBlurbStr,
    ToolCallChainExtension,
    ToolCallExtension,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.shared_params.function_definition import FunctionDefinition
from utils.logger import get_logger

from examples.config import confh

logger = get_logger(__name__)


@CustomStreamHook
def print_each_word_hook(ha: CustomHookArgs) -> None:
    print(ha.head_as_str, end='', flush=True)


@CustomStreamHook
def contradiction_hook(ha: CustomHookArgs) -> None:
    head = ha.head_as_str
    if 'contradict' in head.lower():
        logger.critical('CONTRADICTION MENTIONED!')


@CustomToolHook
def get_weather_hook(
    ha: CustomHookArgs[ChatCompletionChunk, Any],
) -> ToolCallExtension | ToolCallChainExtension[ChatCompletionChunk] | None:
    fn_name = 'get_weather'
    blurb = ToolBlurbStr(f'[used function `{fn_name}`]')

    cm = ha.context_manager
    tool = find_last_tool(ha.chunks)

    if tool is None or tool.function.name != fn_name:
        return None

    with MultiroundContextManager(tmp_from=cm) as cm:
        cm.assistant_message(ha.text, [tool])
        cm.tool_message(
            content='It is 24 degrees Celcius.', tool_call_id=tool.id
        )

        completion = ha.called_by(cm, None)

        # XXX: This is nessecary
        if isinstance(completion, str):
            return blurb, completion
        else:
            return blurb, completion


@CustomToolHook
def get_time_hook(
    ha: CustomHookArgs[ChatCompletionChunk, Any],
) -> ToolCallExtension | ToolCallChainExtension[ChatCompletionChunk] | None:
    fn_name = 'get_time'
    blurb = ToolBlurbStr(f'[used function `{fn_name}`]')

    cm = ha.context_manager
    tool = find_last_tool(ha.chunks)

    if tool is None or tool.function.name != fn_name:
        return None

    with MultiroundContextManager(tmp_from=cm) as cm:
        cm.assistant_message(ha.text, [tool])
        cm.tool_message(
            content=f'It is {datetime.now()}.', tool_call_id=tool.id
        )

        completion = ha.called_by(cm, None)

        # XXX: This is nessecary
        if isinstance(completion, str):
            return blurb, completion
        else:
            return blurb, completion


@CustomCaptureFinish
def capture_finish(chunk: ChatCompletionChunk) -> bool:
    return bool(chunk.choices[0].finish_reason)


llm = LLMWrapper()

example_profile = LLMProfile(
    os.environ[confh.games.api_example_api_key_env_var],
    confh.games.api_example_api_endpoint,
    model=confh.games.api_example_api_model,
    capture_finish=capture_finish,
    system_prompt='You are a helpful assistant.\n'
    'NOTE: Tool call outputs are destroyed on the next tool call. '
    'Please take whatever nessecary before invoking another tool, whether by.'
    'immediately answering the question, or repeating the contents so they are '
    'available in context. This is intentional to limit cache size. ',
    stream_hook=print_each_word_hook | contradiction_hook,
    tool_hook=get_weather_hook | get_time_hook,
)

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

    with MultiroundContextManager() as cm:
        completion = llm.completion(
            'example',
            cm.user_message(
                'What happens when \\sqrt{D} is in the base field in Galois theory?\n'
                "What's he weather in Hangzhou?\n"
                "What's the time in Hangzhou?\n"
                'Do the tool calling limitations in the system prompt make sense?'
            ),
            stream=True,
            tools=tools,
        )

    text = completion.exhaust()

    logger.debug(f'Reasoning:\n{text.reasoning_content}')
    logger.info(f'Content:\n{text.content_with_tool_blurbs}')

    return completion


if __name__ == '__main__':
    example_completion(llm)
