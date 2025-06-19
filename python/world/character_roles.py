#!/usr/bin/env python3
from enum import StrEnum, auto
import os
from typing import Callable, Iterable, Literal, TypeAlias

from configuration.config import ConfigHolder
from llm.api import GameLLM
from llm.utils.context_manager import Context
from llm.utils.openai_api import LLMProfile, ToolsHookPair, capture_finish
from llm.utils.stream import (
    ContentStr,
    CustomHookArgs,
    CustomStreamHook,
    ReasoningContentStr,
)
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from utils.ansi import DARKGREY, END, GREEN
from utils.timestamp import timestamp
from world.character.character_class import Character


# TODO: Move to plot module.
class Plot:
    pass


# TODO: Move to scene module.
class Scene:
    pass


# TODO: Make extendable.
class GameRole(StrEnum):
    CHARACTER = auto()
    ORCHESTRATOR = auto()


@CustomStreamHook
def update_text_hook(
    args: CustomHookArgs[
        ChatCompletionChunk,
        Character,
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    ],
) -> None:
    head = args.head_as_str
    if isinstance(head, ReasoningContentStr):
        print(DARKGREY + args.head_as_str + END, end='', flush=True)
    elif isinstance(head, ContentStr):
        print(GREEN + args.head_as_str + END, end='', flush=True)


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

#############

get_time_hook = GameLLM.create_tool_hook(
    'get_time',
    lambda *_: f'{timestamp.day_of_week_name} '
    f'{timestamp.hour}:{timestamp.minute}:{timestamp.second}, '
    f'{timestamp.day_name} {timestamp.month_name}, '
    f'{timestamp.year}.',
)

get_weather_hook = GameLLM.create_tool_hook(
    'get_weather', lambda *_: 'Not implemented. Make something reasonable up.'
)


class RoleProfiles:
    universal_character_tp = ToolsHookPair(
        tools=[get_time, get_weather],
        hook=get_time_hook | get_weather_hook,
    )

    def __init__(self, confh: ConfigHolder) -> None:
        self.confh = confh

        self.character_profile = LLMProfile(
            'character',
            os.environ[confh.games.api_example_api_key_env_var],
            confh.games.api_example_api_endpoint,
            model=confh.games.api_example_api_model,
            capture_finish=capture_finish,
            stream_hook=update_text_hook,
            tool_hook=self.universal_character_tp.hook,
            completion_kwargs={'tools': self.universal_character_tp.tools},
        )

    def get_handover_prompt(self, next_character: Character) -> str:
        return (
            'It has been determined the next character '
            f'to act is {next_character.name}.'
        )

    def get_user_handover_scratch_prompt(self, character: Character) -> str:
        return (
            f'{character.name}, follow the instructions in your system prompt.'
        )

    def get_user_handover_scratch_prompt_conversation_end(
        self, character: Character
    ) -> str:
        return (
            f'{character.name}, the current conversation has ended. Follow the '
            'instructions in your system prompt.'
        )

    def get_character_system_prompt(
        self, character: Character, directive: str = ''
    ) -> str:
        name = character.name
        # Prompt adapted from [placeholder]
        # TODO: Find placeholder again
        return f"""Write your next reply from {name}'s point of view.
Write how you think {name} would reply based on {name}'s previous messages.
Avoid writing as the other character(s) or Narrator.

DO NOT prefix with `<{name}>:`, the system will add these automatically.

Additional directives: {directive or None}

Today is {timestamp.month_name} {timestamp.day_name}, {timestamp.year}. It is a {timestamp.day_of_week_name}.
"""

    def get_character_scratch_writer_system_prompt(
        self, character: Character, directive: str = ''
    ) -> str:
        name = character.name
        return (
            self.get_character_system_prompt(character, directive)
            + f"""
In addition to the above, you are the ephemeral scratch-writing agent representing {name}.
Based on the previous rounds of conversation, update your scratchpad.
Use this space to reason and come up with plans.
Include:
    - Secrets
    - Thoughts
    - Plans
        - Short term
        - Long term
and whatever else you deem fit.

If the scratchpad does not need changing, follow the tool instructions and omit the `contents` key.
"""
        )

    def get_character_scratch_writer_system_prompt_end(
        self, character: Character, directive: str = ''
    ) -> str:
        name = character.name
        return f"""The current round of conversation has ended.
You are the ephemeral scratch-writing agent representing {name}.
Based on the previous rounds of conversation, update your scratchpad.

Make guesses on when you might meet the other character(s) again.

Additional directives: {directive or None}

Clean up and only keep what will be useful to carry over into future conversations.
"""

    def get_system_system_prompt(self, plot: Plot) -> str:
        return ''


Orchestrator: TypeAlias = Callable[
    [Plot, Context, Iterable[Character]],
    tuple[None, None] | tuple[Character, str],
]


def get_next_round_settings(
    plot: Plot,
    context: Context,
    chars: Iterable[Character],
    cur_char: Character,
):
    next_round = GameLLM.create_tool(
        name='next_round',
        description="""Prepare settings for the next round of conversation.
    Use for:
    1) Next actions/scene plans (i.e. if you decide to lie, don't forget you were lying later on)
    2) Clues/observations (store things that may be relevant in the future here)
    3) Conversation anchors/observations about characters' emotional states (same as above)
    4) Working theories, not permanent facts (don't speak all your theories out loud)
    5) Important tool call results (that you aren't going to immediately speak out loud)
    ALWAYS INCLUDE EXISTING CONTENT YOU WANT TO PRESERVE!""",
        properties={
            'next_speaker': {
                'type': 'string',
                'description': 'New scratch content. '
                'Replaces all previous content.',
            }
        },
        required=['contents'],
    )
    pass
