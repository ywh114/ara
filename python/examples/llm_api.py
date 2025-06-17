#!/usr/bin/env python3
import os
from datetime import datetime
from pprint import pp
from typing import TypeVar

from llm.utils.context_manager import MultiroundContextManager
from llm.utils.openai_api import (
    ChatCompletionToolParams,
    LLMProfile,
)
from llm.utils.stream import (
    ContentStr,
    CustomCaptureFinish,
    CustomHookArgs,
    CustomResponseStr,
    CustomStreamHandler,
    CustomStreamHook,
    ReasoningContentStr,
)
from openai.types.chat import ChatCompletionChunk
from llm.api import GameLLM
from world.character.memory import Memory
from utils.ansi import DARKGREY, END, GREEN
from utils import timestamp
from utils.logger import get_logger

from examples.config import confh
from examples.card import cardh
from examples.character import char

logger = get_logger(__name__)


@CustomStreamHook
def print_each_word_hook(args: CustomHookArgs) -> None:
    head = args.head_as_str
    if isinstance(head, ReasoningContentStr):
        print(DARKGREY + args.head_as_str + END, end='', flush=True)
    elif isinstance(head, ContentStr):
        print(GREEN + args.head_as_str + END, end='', flush=True)


@CustomStreamHook
def cat_hook(ha: CustomHookArgs) -> None:
    if '喵' in ha.head_as_str.lower():
        logger.critical('(ฅ>ω<ฅ)')


T = TypeVar('T')


get_weather_hook = GameLLM.create_tool_hook(
    'get_weather', lambda *_: 'It is 19 degrees Celcius.'
)

get_time_hook = GameLLM.create_tool_hook(
    'get_time', lambda *_: f'It is {datetime.now()}.'
)


@CustomCaptureFinish
def capture_finish(chunk: ChatCompletionChunk) -> bool:
    return bool(chunk.choices[0].finish_reason)


get_weather = GameLLM.create_tool(
    name='get_weather',
    description='Get weather of an location, the user shoud supply a location first',
    properties={
        'location': {
            'type': 'string',
            'description': 'The location, e.g. San Francisco, CA, USA.',
        }
    },
    required=['location'],
)
get_time = GameLLM.create_tool(
    name='get_time',
    description='Get the current time of a location',
    properties={
        'location': {
            'type': 'string',
            'description': 'The location, e.g. San Francisco, CA, USA.',
        }
    },
    required=['location'],
)

tools: ChatCompletionToolParams = [
    {'type': 'function', 'function': get_weather},
    {'type': 'function', 'function': get_time},
    {'type': 'function', 'function': Memory.write_scratch},
]

example_profile = LLMProfile(
    'example',
    os.environ[confh.games.api_example_api_key_env_var],
    confh.games.api_example_api_endpoint,
    model=confh.games.api_example_api_model,
    capture_finish=capture_finish,
    system_prompt=f"""Write your next reply from {{user}}'s point of view. Write how you think {{user}} would reply based on {{user}}'s previous messages. Avoid writing as the character(s) or Narrator.

Do not prefix your messages with <{{user}}>. This is done automatically.

Today is {timestamp.month_name} {timestamp.day_name}, {timestamp.year}. It is {timestamp.day_of_week_name}. 
""".format(user=cardh.get_field('name')),
    stream_hook=print_each_word_hook | cat_hook,
    tool_hook=get_weather_hook | get_time_hook,
    completion_kwargs={'tools': tools},
)

llm = GameLLM(example_profile)


def example_conversation(llm: GameLLM, *, stream: bool = True) -> None:
    cm0 = MultiroundContextManager(char.whoami)
    user_prompt = ''
    while 'exit' not in user_prompt.lower():
        user_prompt = input('User> ')
        with MultiroundContextManager(tmp_from=cm0) as cm1:
            cm1.concat_context(char.scratch)
            cm1.user_message(user_prompt)
            pp(cm1.context)
            completion = llm.completion(
                'example',
                cm1,
                stream=stream,
                specific_tool_hook=char.memory.write_scratch_hook,
            )

        cm0.user_message(user_prompt)
        if stream:
            assert isinstance(completion, CustomStreamHandler)
            cm0.assistant_message(
                completion.exhaust().content_with_tool_blurbs,
                tool_calls=[],
                name=char.cardh.get_field('name'),
            )
            print()
        else:
            assert isinstance(completion, CustomResponseStr)
            cm0.assistant_message(
                completion.content_with_tool_blurbs,
                tool_calls=[],
                name=char.cardh.get_field('name'),
            )


if __name__ == '__main__':
    example_conversation(llm, stream=True)
