#!/usr/bin/env python3
############################################################################
#                                                                          #
#  Copyright (C) 2025                                                      #
#                                                                          #
#  This program is free software: you can redistribute it and/or modify    #
#  it under the terms of the GNU General Public License as published by    #
#  the Free Software Foundation, either version 3 of the License, or       #
#  (at your option) any later version.                                     #
#                                                                          #
#  This program is distributed in the hope that it will be useful,         #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#  GNU General Public License for more details.                            #
#                                                                          #
#  You should have received a copy of the GNU General Public License       #
#  along with this program. If not, see <http://www.gnu.org/licenses/>.    #
#                                                                          #
############################################################################
from pprint import pp
from typing import Any, Callable, Hashable, NamedTuple, Self, TypeVar

from llm.utils.context_manager import (
    MultiroundContextManager,
)
from utils.logger import get_logger
from world.character.character_class import Character
from world.character_roles import (
    GameRole,
    NextSceneChoice,
    Orchestrator,
    Plot,
    RoleProfiles,
    get_next_round_settings,
)
from world.importance import CharacterImportance
from world.plot import Location, PlotMarcher

logger = get_logger(__name__)


T = TypeVar('T', bound=Hashable)


class RPInfo(NamedTuple):
    """
    Container for conversation state information during roleplay sessions.

    :param rp: Role configuration profiles for game characters.
    :param plot: Current narrative progression handler.
    :param cm: Multiround context manager instance.
    :param char_pool: Character pool.
    :param next_char: Character scheduled to act in current round.
    :param player_char: Player-controlled character reference.
    :param narrator_char: Narrator character reference.
    :param here_chars: Characters currently present in scene.
    :param away_chars: Characters currently absent from scene.
    :param entering_chars: Characters entering.
    :param exiting_chars: Characters exiting.
    :param loc_pool: Location pool.
    :param next_loc: Location of the next round.
    :param directive: Current narrative instruction for character actions.
    :param suggestions: Suggested dialogue/action options for player.
    :param get_user_prompt: Callback to generate player prompt text.
    :param get_user_debug: Callback for debug information display.
    """

    rp: RoleProfiles
    plot: Plot
    cm: MultiroundContextManager
    char_pool: set[Character]
    next_char: Character
    player_char: Character
    narrator_char: Character
    here_chars: set[Character]
    away_chars: set[Character]
    entering_chars: set[Character]
    exiting_chars: set[Character]
    loc_pool: set[Location]
    next_loc: Location
    directive: str
    suggestions: list[str]
    get_user_prompt: Callable[[Self], str]
    get_user_debug: Callable[[Self, Any], None]


def narrator_action(i: RPInfo) -> MultiroundContextManager:
    """
    Process narrator turn in conversation sequence.

    :param i: Current conversation state container.
    :returns: Updated context manager with narrator response.
    """
    # If the next actor is the narrator, pad and add.
    logger.info(f'{i.next_char.name} [Narrator] is speaking now.')
    with MultiroundContextManager(tmp_from=i.cm) as cm1:
        cm1.pad_context_for('scratch')
        cm1.concat_context(i.next_loc.scratch)
        completion = i.rp.llm.completion(
            GameRole.NARRATOR,
            i.rp.get_narrator_system_prompt(
                player_char=i.player_char,
                narrator_char=i.narrator_char,
                directive=i.directive,
                here_chars=i.here_chars,
                away_chars=i.away_chars,
                plot=i.plot,
            ),
            cm1,
            stream=True,
        )

    if True or __debug__:  # TODO: remove.
        # Force a newline.
        print()

    # Write to non-transient context.
    i.cm.pad_context_for('assistant', base=True)
    i.cm.assistant_message(
        completion.exhaust().content,
        tool_calls=[],
        name=f'{i.narrator_char.name}',
    )

    return i.cm


def player_action(i: RPInfo) -> MultiroundContextManager:
    """
    Process player turn in conversation sequence.

    :param i: Current conversation state container.
    :returns: Updated context manager with player input.
    """
    logger.info(f'{i.next_char.name} [Player] is speaking now.')
    logger.info(i.next_loc.desc)
    # If next actor is player, pad and add.
    i.cm.pad_context_for('user', base=True)
    i.cm.user_message(
        i.get_user_prompt(i),
        name=i.player_char.name,
    )

    return i.cm


def character_action(i: RPInfo) -> MultiroundContextManager:
    """
    Process NPC turn in conversation sequence.

    :param i: Current conversation state container.
    :returns: Updated context manager with character response.
    """
    logger.info(f'{i.next_char.name} is speaking now.')
    # If next actor is LLM, pad and call.
    # Get character response.
    with MultiroundContextManager(
        injected_context=i.next_char.whoami, tmp_from=i.cm
    ) as cm1:
        # Set context to what the next character should see.
        # Character is present so pad order should not change.
        cm1.filter_to(i.next_char)
        if i.next_char.importance < CharacterImportance.IMPORTANT:
            cm1.pad_context_for('assistant')
        else:
            # Pad and dump scratch.
            if False:
                cm1.pad_context_for('scratch')
                cm1.concat_context(
                    i.rp.get_location_and_character_scratch(
                        i.next_loc, i.next_char
                    )
                )
            else:
                cm1.pad_context_for('scratch')
                cm1.concat_context(i.next_loc.scratch)

        completion = i.rp.llm.completion(
            GameRole.CHARACTER,
            i.rp.get_character_system_prompt(i.next_char, i.directive, i.plot),
            cm1,
            stream=True,
        )

    # Write to non-transient context.
    i.cm.pad_context_for('assistant', base=True)
    i.cm.assistant_message(
        completion.exhaust().content,
        tool_calls=[],
        name=i.next_char.name,
    )

    if True or __debug__:  # TODO: remove.
        # Force a newline.
        print()

    if False and i.next_char.importance >= CharacterImportance.IMPORTANT:
        # Let characters update their scratch.
        # A two-pass approach seems to be required to maintain
        # consistency.
        with MultiroundContextManager(
            injected_context=i.next_char.whoami, tmp_from=i.cm
        ) as cm1:
            # Set context to what the next character should see.
            # TODO: Required characters should have vectorized scratch/self
            # modification chance.
            # Important characters should have scratch modification change.
            cm1.filter_to(i.next_char)
            # Dump handover message.
            cm1.pad_context_for(
                'assistant',
                content=i.rp.get_user_handover_scratch_prompt(i.next_char),
                padding_name=MultiroundContextManager.default_sysname,
            )
            i.rp.llm.completion(
                GameRole.CHARACTER,
                i.rp.get_character_system_prompt(
                    i.next_char, i.directive, i.plot
                ),
                cm1,
                stream=True,
                specific_tools=i.next_char.memory.chat_tools_end.toolparams,
                specific_tool_hook=i.next_char.memory.chat_tools_end.hook,
                tool_choice='required',
            ).exhaust()

    return i.cm


def flow(hi: set[T], lo: set[T], /, s: set[T]) -> tuple[set[T], set[T]]:
    """
    Transfer elements between high/low priority sets.

    :param hi: The source.
    :param lo: The destination.
    :param s: Elements to transfer from `hi` to `lo`.
    :returns: Updated `(hi, lo)` tuple after transfer.
    """
    return hi - s, lo | s


def multiround_conversation(
    rp: RoleProfiles,
    pm: PlotMarcher,
    /,
    player_char: Character,
    narrator_char: Character,
    *,
    get_user_prompt: Callable[[RPInfo], str],
    get_user_debug: Callable[[RPInfo, Any], None],
    call_orchestrator: Orchestrator = get_next_round_settings,
) -> None:
    """
    Execute multi-round character-driven conversation sequence.

    :param rp: Role configuration profiles.
    :param plot: Narrative progression handler.
    :param player_char: Player-controlled character.
    :param narrator_char: Narrator character.
    :param get_user_prompt: Callback for player prompt generation.
    :param get_user_debug: Callback for debug information display.
    :param call_orchestrator: Orchestrator for round sequencing logic.
    """
    plot = pm.plot
    char_pool = plot.character_pool
    away_chars, here_chars = flow(char_pool, set(), s=plot.starting_characters)
    directives_log = dict.fromkeys(char_pool, '')

    loc_pool = plot.location_pool
    loc = plot.starting_location
    prev_char = None

    with MultiroundContextManager(
        *char_pool,
    ) as cm0:
        cm0.enter_entities(*here_chars)
        logger.debug(
            f'Entered conversation with {[char.name for char in here_chars]}. '
            f'Off-scene characters: {[char.name for char in away_chars]}'
        )

        info: RPInfo | None = None
        while not isinstance(
            maybe_quintuple := call_orchestrator(
                rp,
                plot,
                cm0,
                here_chars,
                away_chars,
                prev_char,
                player_char,
                narrator_char,
                loc_pool,
                loc,
            ),
            NextSceneChoice,
        ):
            # Unpack results.
            (
                next_char,
                directive,
                suggestions,
                entering_chars,
                exiting_chars,
                switch_loc,
                edit_loc,
            ) = maybe_quintuple

            prev_char = next_char

            if edit_loc:
                logger.debug(f'Edit location environment: {edit_loc}')
                pass

            if switch_loc:
                logger.debug(f'Switch location: {loc} -> {switch_loc}')
                if loc == switch_loc:
                    logger.warning(f'Already at {loc}.')
                loc = switch_loc

            # Pack info.
            info = RPInfo(
                rp=rp,
                plot=plot,
                cm=cm0,
                char_pool=char_pool,
                next_char=next_char,
                player_char=player_char,
                narrator_char=narrator_char,
                here_chars=here_chars,
                away_chars=away_chars,
                entering_chars=entering_chars,
                exiting_chars=exiting_chars,
                loc_pool=loc_pool,
                next_loc=loc,
                directive=directive,
                suggestions=suggestions,
                get_user_prompt=get_user_prompt,
                get_user_debug=lambda *_: None,
            )

            # Log directive.
            logger.debug(
                f'Speaker: {next_char.name}, directive: {directive}, '
                f'suggestions: {suggestions}'
            )

            logger.debug(
                'Due to enter: '
                f'{[c.name for c in entering_chars]}, '
                'due to exit: '
                f'{[c.name for c in exiting_chars]}'
            )
            directives_log[next_char] = directive

            if __debug__:
                # Wait for debug.
                get_user_debug(info, '')

            # Entering characters enter immediately.
            if entering_chars:
                logger.debug(f'Enter: {[c.name for c in entering_chars]}')
                cm0.enter_entities(*entering_chars)
                away_chars, here_chars = flow(
                    away_chars, here_chars, entering_chars
                )

            # Perform actions.
            if next_char == narrator_char:
                narrator_action(info)
            elif next_char == player_char:
                player_action(info)
            else:
                character_action(info)

            # Exiting characters leave now.
            if exiting_chars:
                logger.debug(f'Exit: {[c.name for c in exiting_chars]}')
                cm0.exit_entities(*exiting_chars)
                here_chars, away_chars = flow(
                    here_chars, away_chars, exiting_chars
                )

        # Point to next scene.
        pm.elect(maybe_quintuple)
        next_plot = next(pm)

        next_char_pool = next_plot.character_pool
        next_starting_chars = next_plot.starting_characters
        next_loc_pool = next_plot.character_pool
        next_starting_loc = next_plot.starting_location
        # TODO: logic

        logger.debug('Conversation ended.')
        pp(cm0.seen_entities)
        pp(
            [
                c
                for c in cm0.seen_entities
                if c != player_char
                if c != narrator_char
                if c.importance >= CharacterImportance.IMPORTANT
            ]
        )
        # After final _exit__().
        # FIXME: Convoluted logic.
        # TODO: Make this async by adding character conversation lock.
        # TODO: Consider using vectorized tool call for few characters.
        # TODO: - store into db
        for char in (
            c
            for c in cm0.seen_entities
            if c != player_char
            if c != narrator_char
            if c.importance >= CharacterImportance.IMPORTANT
        ):
            # Exit injections.
            # TODO: Perform exit actions based on importance:
            # - store into character memory db
            # - store into character knowledge db
            # - morph characters if required
            # - lift/lower character importance if required
            with MultiroundContextManager(
                injected_context=char.whoami, tmp_from=cm0
            ) as cm1:
                # Set context to what the next character should see.
                cm1.filter_to(char)
                # Dump scratch.
                cm1.pad_context_for('scratch')
                if False:
                    cm1.concat_context(
                        rp.get_location_and_character_scratch(loc, char)
                    )
                # Re-pad with instructions.
                cm1.pad_context_for(
                    'assistant',
                    content=rp.get_user_handover_scratch_prompt_conversation_end(
                        char
                    ),
                    padding_name=MultiroundContextManager.default_sysname,
                )

                rp.llm.completion(
                    GameRole.CHARACTER,
                    rp.get_character_scratch_writer_system_prompt_end(
                        char, directives_log[char], plot
                    ),
                    cm1,
                    stream=True,
                    specific_tools=char.memory.chat_tools_end.toolparams,
                    specific_tool_hook=char.memory.chat_tools_end.hook,
                    tool_choice='required',
                ).exhaust()

    logger.debug('Exited conversation.')
    if info is not None:
        get_user_debug(info, '')
