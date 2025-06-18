#!/usr/bin/env python3
from pprint import pp
from typing import Callable, Iterable

from llm.utils.context_manager import (
    Context,
    MultiroundContextManager,
    RegisterHook,
)
from python.llm.api import GameLLM
from python.utils.logger import get_logger
from python.world.character.memory import Memory
from world.character.character_class import Character

logger = get_logger(__name__)


@RegisterHook
def example_register_hook(x: Character, y: Context):
    pp(x)
    pp(y)


def new_conversation(
    llm: GameLLM,
    /,
    *starting_chars: Character,
    player_char: Character,
    get_user_prompt: Callable[..., str],
    get_next_speaker: Callable[
        [Context, Iterable[Character]], Character | None
    ],
):
    chars = starting_chars
    with MultiroundContextManager(
        *chars,
        register_hook=example_register_hook,
    ) as cm0:
        logger.debug(
            f'Entered conversation with {[char.name for char in chars]}.'
        )
        next_prompt = ''
        while True:
            next_speaker = get_next_speaker(cm0.context, chars)

            if next_speaker is None:
                break

            if _next_is_player := next_speaker == player_char:
                next_prompt = get_user_prompt()
                next_name = player_char.name
            else:
                next_prompt = 'You are the next to speak.'
                next_name = 'System'
            # First round of injections.
            with MultiroundContextManager(
                injected_context=next_speaker.whoami, tmp_from=cm0
            ) as cm1:
                cm1.concat_context(next_speaker.scratch)
                cm1.user_message(next_prompt, name=next_name)
                pp(cm1.context)
                completion = llm.completion(
                    'example',
                    cm1,
                    stream=True,
                )

            # Save output.
            cm0.user_message(
                next_prompt if _next_is_player else '',
                name=next_name if _next_is_player else '',
            )
            cm0.assistant_message(
                completion.exhaust().content_with_tool_blurbs,
                tool_calls=[],
                name=next_speaker.name,
            )

            print()  # XXX: Remove later.

            if not _next_is_player:
                # Second round of injections.
                with MultiroundContextManager(
                    injected_context=next_speaker.whoami, tmp_from=cm0
                ) as cm1:
                    cm1.user_message(
                        'Based on the previous round of conversation, '
                        'update your scratchpad.\n'
                        'Use this space to reason and come up with plans.\n'
                        'Include all your secrets, thoughts and plans. '
                        'Stay in character!',
                        name='System',
                    )
                    completion = llm.completion(
                        'example',
                        cm1,
                        stream=True,
                        specific_tools=[
                            {
                                'type': 'function',
                                'function': Memory.write_scratch,
                            }
                        ],
                        specific_tool_hook=next_speaker.memory.write_scratch_hook_end,
                        tool_choice='required',
                    )
                    completion.exhaust()
    logger.debug('Exited conversation.')
