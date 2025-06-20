#!/usr/bin/env python3
from pprint import pp
from typing import Callable, Iterable

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
    get_next_round_settings,
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
    rp: RoleProfiles,
    plot: Plot,
    /,
    player_char: Character,
    narrator_char: Character,
    *,
    get_user_prompt: Callable[[Iterable[str]], str],
    call_orchestrator: Orchestrator = get_next_round_settings,
):
    char_pool = plot.character_pool
    chars = plot.starting_characters
    off_scene_chars = char_pool - chars
    directives_log = dict.fromkeys(char_pool, '')

    with MultiroundContextManager(
        *char_pool,
        register_hook=example_register_hook,
    ) as cm0:
        logger.debug(
            f'Entered conversation with {[char.name for char in chars]}. '
            f'Off-scene characters: {[char.name for char in off_scene_chars]}'
        )
        cm0.enter_entities(*chars)

        while (
            tup5 := call_orchestrator(
                rp,
                plot,
                cm0,
                chars,
                off_scene_chars,
                player_char,
                narrator_char,
            )
        ) is not None:
            # Get orchestrator instructions.
            next_char, directive, suggestions, entering_chars, exiting_chars = (
                tup5
            )

            # Entering characters enter immediately.
            if entering_chars:
                cm0.enter_entities(*entering_chars)
                chars |= entering_chars
                off_scene_chars -= entering_chars
                logger.debug(f'Enter: {[c.name for c in entering_chars]}')

            # Determine order for padding.
            prev_was_player = (
                cm0.head is not None and cm0.head['role'] == 'user'
            )
            next_is_player = next_char == player_char
            next_is_narrator = next_char == narrator_char

            # Log directive.
            directives_log[next_char] = directive

            # XXX: ## Start of conversation section. ###
            if next_is_narrator:
                # If the next actor is the narrator, pad and add.
                with MultiroundContextManager(tmp_from=cm0) as cm1:
                    if not prev_was_player:
                        # Pad with handover prompt if assistant acts twice.
                        cm1.user_message(
                            rp.get_handover_narrator_prompt(narrator_char),
                            name='System',
                            suppress_decorations=True,
                        )
                    completion = rp.llm.completion(
                        GameRole.NARRATOR,
                        rp.get_narrator_system_prompt(
                            player_char,
                            narrator_char,
                            directives_log[next_char],
                            plot,
                        ),
                        cm1,
                        stream=True,
                    )
                # Write to non-transient context.
                if not prev_was_player:
                    # Pad with handover prompt if assistant acts twice.
                    cm0.user_message(
                        rp.get_handover_narrator_prompt(narrator_char),
                        name='System',
                        suppress_decorations=True,
                    )
                cm0.assistant_message(
                    completion.exhaust().content_with_tool_blurbs,
                    tool_calls=[],
                    name=f'{narrator_char.name} [Narrator]',
                )
            elif next_is_player:
                # If next actor is player, pad and add.
                if prev_was_player:
                    # Pad handover `assistant_message` in case user acts twice.
                    cm0.assistant_message(
                        rp.get_handover_prompt(player_char),
                        tool_calls=[],
                        name='System',
                        suppress_decorations=True,
                    )
                cm0.user_message(
                    get_user_prompt(suggestions),
                    name=player_char.name,
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
                    # TODO: Consider different format for `char.scratch`
                    # for unified padding.
                    cm1.filter_to(next_char)
                    # Pad for space to dump scratch if previous message came
                    # from the player (scratch is ->user->assistant).
                    if prev_was_player:
                        cm1.assistant_message(
                            rp.get_handover_prompt(next_char),
                            tool_calls=[],
                            name='System',
                            suppress_decorations=True,
                        )
                    # Dump scratch.
                    cm1.concat_context(next_char.scratch)

                    completion = rp.llm.completion(
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
                        suppress_decorations=True,
                    )
                cm0.assistant_message(
                    completion.exhaust().content_with_tool_blurbs,
                    tool_calls=[],
                    name=next_char.name,
                )

                print()  # FIXME: Remove later.

                # Let characters update their scratch.
                # A two-pass approach seems to be required to maintain
                # consistency.
                with MultiroundContextManager(
                    injected_context=next_char.whoami, tmp_from=cm0
                ) as cm1:
                    # Set context to what the next character should see.
                    cm1.filter_to(next_char)
                    # Pad with handover message.
                    cm1.user_message(
                        rp.get_user_handover_scratch_prompt(next_char),
                        name='System',
                        suppress_decorations=True,
                    )
                    rp.llm.completion(
                        GameRole.CHARACTER,
                        rp.get_character_system_prompt(
                            next_char, directives_log[next_char]
                        ),
                        cm1,
                        stream=True,
                        specific_tools=next_char.memory.chat_tools_end.toolparams,
                        specific_tool_hook=next_char.memory.chat_tools_end.hook,
                        tool_choice='required',
                    ).exhaust()
            # XXX:## End of conversation section. ###

            # Exiting characters leave now.
            if exiting_chars:
                cm0.exit_entities(*exiting_chars)
                chars -= exiting_chars
                off_scene_chars |= exiting_chars
                logger.debug(f'Exit: {[c.name for c in exiting_chars]}')

        logger.debug('Conversation ended.')
        # TODO: Make this async by adding character conversation lock.
        # TODO: Consider using vectorized tool call for few characters.
        for char in cm0.seen_entities:
            if char == player_char or char == narrator_char:
                continue
            # Exit injections.
            with MultiroundContextManager(
                injected_context=char.whoami, tmp_from=cm0
            ) as cm1:
                # Set context to what the next character should see.
                cm1.filter_to(char)
                # Padding.
                # TODO: name=Padding, cm.pad_assistant/cm.pad_user
                if cm1.head is not None and cm1.head['role'] == 'user':
                    cm1.assistant_message(
                        '',
                        tool_calls=[],
                        name='System',
                        suppress_decorations=True,
                    )
                # Dump scratch.
                cm1.concat_context(char.scratch)
                # Pad with handover/end message.
                cm1.user_message(
                    rp.get_user_handover_scratch_prompt_conversation_end(char),
                    name='System',
                )

                rp.llm.completion(
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
