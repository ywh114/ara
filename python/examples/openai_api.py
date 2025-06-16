#!/usr/bin/env python3
import os
from datetime import datetime
from typing import Any, Callable, TypeAlias, TypeVar

from llm.utils.context_manager import Context, MultiroundContextManager
from llm.utils.openai_api import (
    ChatCompletionToolParams,
    LLMProfile,
    LLMWrapper,
)
from llm.utils.stream import (
    CustomCaptureFinish,
    CustomHookArgs,
    CustomStreamHook,
    CustomToolHook,
    ToolBlurbStr,
    ToolCallChainExtension,
    ToolCallExtension,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageToolCall
from openai.types.shared_params.function_definition import FunctionDefinition
from utils import timestamp
from utils.logger import get_logger

from examples.config import confh
from examples.card import cardh

logger = get_logger(__name__)


@CustomStreamHook
def print_each_word_hook(ha: CustomHookArgs) -> None:
    print(ha.head_as_str, end='', flush=True)


@CustomStreamHook
def cat_hook(ha: CustomHookArgs) -> None:
    if '喵' in ha.head_as_str.lower():
        logger.critical('(ฅ´ω`ฅ)')


T = TypeVar('T')

CustomToolHookContentsFn: TypeAlias = Callable[
    [CustomHookArgs[ChatCompletionChunk, Any]], str
]


def create_tool_hook(
    fn_name: str, tool_hook_contents_fn: CustomToolHookContentsFn
) -> CustomToolHook[ChatCompletionChunk, Any, ChatCompletionMessageToolCall]:
    blurb = ToolBlurbStr(
        f'<SYSTEM - AUTOMATICALLY INJECTED - DO NOT IMITATE: used {fn_name}>'
    )

    @CustomToolHook
    def custom_tool_hook(
        ha: CustomHookArgs[ChatCompletionChunk, Any],
        _tools: list[ChatCompletionMessageToolCall],
    ) -> ToolCallExtension | ToolCallChainExtension[ChatCompletionChunk] | None:
        cm = ha.context_manager

        tools = tuple(tool for tool in _tools if tool.function.name == fn_name)

        if not tools:
            return None

        assert len(tools) == 1
        tool = tools[0]

        content = tool_hook_contents_fn(ha)

        cm.assistant_message(ha.text.content_with_reasoning, [tool])
        cm.tool_message(content=content, tool_call_id=tool.id)

        completion = ha.called_by(cm, None)

        # XXX: This is nessecary
        if isinstance(completion, str):
            return blurb, completion
        else:
            return blurb, completion

    return custom_tool_hook


get_weather_hook = create_tool_hook(
    'get_weather', lambda _: 'It is 32 degrees Celcius.'
)

get_time_hook = create_tool_hook(
    'get_time', lambda _: f'It is {datetime.now()}.'
)


@CustomCaptureFinish
def capture_finish(chunk: ChatCompletionChunk) -> bool:
    return bool(chunk.choices[0].finish_reason)


llm = LLMWrapper()

example_profile = LLMProfile(
    os.environ[confh.games.api_example_api_key_env_var],
    confh.games.api_example_api_endpoint,
    model=confh.games.api_example_api_model,
    capture_finish=capture_finish,
    system_prompt=f"""Write your next reply from {{user}}'s point of view. Write how you think {{user}} would reply based on {{user}}'s previous messages. Avoid writing as the character(s) or Narrator.

Today is {timestamp.month_name} {timestamp.day_name}, {timestamp.year}. It is {timestamp.day_of_week_name}. 
""".format(user=cardh.get_field('name')),
    stream_hook=print_each_word_hook | cat_hook,
    tool_hook=get_weather_hook | get_time_hook,
)

llm['example'] = example_profile

get_weather: FunctionDefinition = {
    'name': 'get_weather',
    'description': 'Get weather of an location, the user shoud supply a location first',
    'parameters': {
        'type': 'object',
        'properties': {
            'location': {
                'type': 'string',
                'description': 'The location, e.g. San Francisco, CA, USA.',
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
                'description': 'The location, e.g. San Francisco, CA, USA.',
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

injected_context: Context = [
    {'role': 'user', 'content': 'Please provide your `name`.'},
    {'role': 'assistant', 'content': cardh.get_field('name')},
    {'role': 'user', 'content': 'Please provide your `summary`.'},
    {'role': 'assistant', 'content': cardh.get_field('summary')},
    {'role': 'user', 'content': 'Please provide your `personality`.'},
    {'role': 'assistant', 'content': cardh.get_field('personality')},
    {'role': 'user', 'content': 'Please provide your `scenario`.'},
    {'role': 'assistant', 'content': cardh.get_field('scenario')},
    {
        'role': 'user',
        'content': 'Please provide your `greeting_message` field.',
    },
    {'role': 'assistant', 'content': cardh.get_field('greeting_message')},
    {
        'role': 'user',
        'content': 'Please provide your `example_messages` field.',
    },
    {'role': 'assistant', 'content': cardh.get_field('example_messages')},
]


def example_completion(
    llm: LLMWrapper,
) -> None:
    context_manager = MultiroundContextManager(injected_context)
    user_prompt = ''
    while user_prompt != 'exit':
        user_prompt = input('User> ')
        context_manager.user_message(user_prompt)
        with MultiroundContextManager(tmp_from=context_manager) as cm:
            completion = llm.completion(
                'example',
                cm,
                stream=True,
                tools=tools,
            )

        context_manager.assistant_message(
            completion.exhaust().content_with_tool_blurbs, tool_calls=[]
        )
        print()


if __name__ == '__main__':
    example_completion(llm)
