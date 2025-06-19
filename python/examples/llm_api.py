#!/usr/bin/env python3
import os
from datetime import datetime
from pprint import pp
from typing import TypeVar

from llm.utils.context_manager import (
    Context,
    MultiroundContextManager,
    RegisterHook,
)
from llm.utils.openai_api import (
    LLMProfile,
    ToolsHookPair,
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
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionToolParam,
)
from llm.api import GameLLM
from world.character_roles import RoleProfiles
from world.character.character_class import Character
from utils.ansi import DARKGREY, END, GREEN
from utils.logger import get_logger

from examples.config import confh
from examples.character import char

logger = get_logger(__name__)


rp = RoleProfiles(confh)


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

tools: list[ChatCompletionToolParam] = [
    {'type': 'function', 'function': get_weather},
    {'type': 'function', 'function': get_time},
]

universal_character_tp = ToolsHookPair(
    tools=[get_weather, get_time],
    hook=get_weather_hook | get_time_hook,
)

example_profile = LLMProfile(
    'example',
    os.environ[confh.games.api_example_api_key_env_var],
    confh.games.api_example_api_endpoint,
    model=confh.games.api_example_api_model,
    capture_finish=capture_finish,
    stream_hook=print_each_word_hook | cat_hook,
    tool_hook=universal_character_tp.hook,
    completion_kwargs={'tools': universal_character_tp.toolparams},
)

llm = GameLLM(example_profile)


@RegisterHook
def example_register_hook(x: Character, y: Context):
    pp(x)
    pp(y)


def example_conversation(llm: GameLLM, *, stream: bool = True) -> None:
    with MultiroundContextManager(
        char,
        register_hook=example_register_hook,
    ) as cm0:
        user_prompt = ''
        while True:
            user_prompt = input('User> ')
            exit_round = 'exit' in user_prompt.lower()

            # First round of injections.
            with MultiroundContextManager(
                injected_context=char.whoami, tmp_from=cm0
            ) as cm1:
                cm1.concat_context(char.scratch)
                cm1.user_message(user_prompt)
                pp(cm1.context)
                completion = llm.completion(
                    'example',
                    rp.get_character_system_prompt(char, ''),
                    cm1,
                    stream=stream,
                    specific_tools=char.memory.chat_tools.toolparams,
                    specific_tool_hook=char.memory.chat_tools.hook,
                )

            # Save output.
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

            if exit_round:
                break

            # Second round of injections.
            with MultiroundContextManager(
                injected_context=char.whoami, tmp_from=cm0
            ) as cm1:
                cm1.concat_context(char.scratch)
                cm1.user_message(
                    rp.get_user_handover_scratch_prompt(char),
                    name='System',
                )
                completion = llm.completion(
                    'example',
                    rp.get_character_scratch_writer_system_prompt(char, ''),
                    cm1,
                    stream=stream,
                    specific_tools=char.memory.chat_tools_end.toolparams,
                    specific_tool_hook=char.memory.chat_tools_end.hook,
                    tool_choice='required',
                )

                if stream:
                    assert isinstance(completion, CustomStreamHandler)
                    completion.exhaust()

        pp('END OF CONVERSATION 0')
        pp(char.memory.scratch)

        # Exit injections.
        with MultiroundContextManager(
            injected_context=char.whoami, tmp_from=cm0
        ) as cm1:
            cm1.concat_context(char.scratch)
            cm1.user_message(
                rp.get_user_handover_scratch_prompt_conversation_end(char),
                name='System',
            )
            completion = llm.completion(
                'example',
                rp.get_character_scratch_writer_system_prompt_end(char, ''),
                cm1,
                stream=stream,
                specific_tools=char.memory.chat_tools_end.toolparams,
                specific_tool_hook=char.memory.chat_tools_end.hook,
                tool_choice='required',
            )

            if stream:
                assert isinstance(completion, CustomStreamHandler)
                completion.exhaust()

        pp('END OF CONVERSATION 1')
        pp(char.memory.scratch)


if __name__ == '__main__':
    example_conversation(llm, stream=True)
