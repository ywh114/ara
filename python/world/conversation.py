#!/usr/bin/env python3
from pprint import pp
from typing import Callable, Iterable, reveal_type

from llm.api import GameLLM
from llm.utils.context_manager import (
    Context,
    MultiroundContextManager,
    RegisterHook,
)
from world.character_roles import (
    GameRole,
    Orchestrator,
    Plot,
    RoleProfiles,
)
from utils.logger import get_logger
from world.character.character_class import Character

logger = get_logger(__name__)


@RegisterHook
def example_register_hook(x: Character, y: Context):
    pp(x)
    pp(y)


# TODO: put 'System' somewhere.
def multiround_conversation(
    llm: GameLLM,
    rp: RoleProfiles,
    plot: Plot,
    /,
    *starting_chars: Character,
    player_char: Character,
    get_user_prompt: Callable[[str], str],
    call_orchestrator: Callable[
        [Plot, Context, Iterable[Character]],
        tuple[None, None] | tuple[Character, str],
    ],
):
    chars = starting_chars
    directives_log = dict.fromkeys(chars, '')

    with MultiroundContextManager(
        *chars,
        register_hook=example_register_hook,
    ) as cm0:
        logger.debug(
            f'Entered conversation with {[char.name for char in chars]}.'
        )

        while None not in (tup := call_orchestrator(plot, cm0.context, chars)):
            next_char, directive = tup
            assert next_char is not None
            assert directive is not None

            directives_log[next_char] = directive

            prev_was_player = (
                cm0.head is not None and cm0.head['role'] == 'user'
            )
            next_is_player = next_char == player_char

            if next_is_player:
                # If next actor is player, pad and add.
                if prev_was_player:
                    # Pad handover `assistant_message` in case user acts twice.
                    cm0.assistant_message(
                        rp.get_handover_prompt(next_char),
                        tool_calls=[],
                        name='System',
                    )
                cm0.user_message(
                    get_user_prompt(directives_log[next_char]),
                    name=next_char.name,
                )
            else:
                # If next actor is LLM, pad and call.
                # Get character response.
                with MultiroundContextManager(
                    injected_context=next_char.whoami, tmp_from=cm0
                ) as cm1:
                    # Set context to what the next character should see.
                    # Character is present so pad order should not change.
                    # TODO: Provide hook for additional custom filters.
                    cm1.filter_to(next_char)
                    # Pad for space to dump scratch if previous message came
                    # from the player (scratch is ->user->assistant).
                    if prev_was_player:
                        cm1.user_message(
                            rp.get_handover_prompt(next_char),
                            name='System',
                        )
                    # Dump scratch.
                    cm1.concat_context(next_char.scratch)

                    completion = llm.completion(
                        GameRole.CHARACTER,
                        rp.get_character_system_prompt(
                            next_char, directives_log[next_char]
                        ),
                        cm1,
                        stream=True,
                    )

                # Write to non-transient context.
                if not prev_was_player:
                    # Pad with handover prompt if assistant acts twice.
                    cm0.user_message(
                        rp.get_handover_prompt(next_char),
                        name='System',
                    )
                cm0.assistant_message(
                    completion.exhaust().content_with_tool_blurbs,
                    tool_calls=[],
                    name=next_char.name,
                )

                print()  # XXX: Remove later.

                # Let characters update their scratch.
                with MultiroundContextManager(
                    injected_context=next_char.whoami, tmp_from=cm0
                ) as cm1:
                    # Set context to what the next character should see.
                    cm1.filter_to(next_char)
                    # Pad with handover message.
                    cm1.user_message(
                        rp.get_user_handover_scratch_prompt(next_char),
                        name='System',
                    )
                    completion = llm.completion(
                        GameRole.CHARACTER,
                        rp.get_character_system_prompt(
                            next_char, directives_log[next_char]
                        ),
                        cm1,
                        stream=True,
                        specific_tools=next_char.memory.chat_tools_end.toolparams,
                        specific_tool_hook=next_char.memory.chat_tools_end.hook,
                        tool_choice='required',
                    )
                    completion.exhaust()

        logger.debug('Conversation ended.')
        # TODO: Make this async by adding character conversation lock.
        # TODO: Consider using vectorized tool call for few characters.
        for char in cm0.seen_entities:
            if char == player_char:
                continue
            # Exit injections.
            with MultiroundContextManager(
                injected_context=char.whoami, tmp_from=cm0
            ) as cm1:
                # Set context to what the next character should see.
                cm1.filter_to(char)
                # Dump scratch.
                cm1.concat_context(char.scratch)
                # Pad with handover/end message.
                cm1.user_message(
                    rp.get_user_handover_scratch_prompt_conversation_end(char),
                    name='System',
                )
                llm.completion(
                    GameRole.CHARACTER,
                    rp.get_character_scratch_writer_system_prompt_end(
                        char, directives_log[char]
                    ),
                    cm1,
                    stream=True,
                    specific_tools=char.memory.chat_tools_end.toolparams,
                    specific_tool_hook=char.memory.chat_tools_end.hook,
                    tool_choice='required',
                ).exhaust()

    logger.debug('Exited conversation.')
