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
import json
import os
from enum import StrEnum, auto
from typing import (
    Any,
    Callable,
    Iterable,
    Protocol,
    TypeAlias,
    TypeVar,
    TypedDict,
)

from configuration.config import ConfigHolder
from llm.api import GameLLM
from llm.utils.context_manager import Context, MultiroundContextManager
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
from utils.ansi import BLUE, END, GREEN
from utils.logger import get_logger
from utils.timestamp import timestamp
from world.character.character_class import Character
from world.plot import Location, Plot

logger = get_logger(__name__)


# TODO: Move to scene module.
class Scene:
    pass


# TODO: Make extendable.
class GameRole(StrEnum):
    CHARACTER = auto()
    NARRATOR = auto()
    ORCHESTRATOR = auto()


@CustomStreamHook
def update_text_hook(  # XXX: DEMO ONLY
    args: CustomHookArgs[
        ChatCompletionChunk,
        Character,
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    ],
) -> None:
    head = args.head_as_str
    if isinstance(head, ReasoningContentStr):
        print(BLUE + args.head_as_str + END, end='', flush=True)
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
            GameRole.CHARACTER,
            os.environ[confh.games.api_example_api_key_env_var],
            confh.games.api_example_api_endpoint,
            model=confh.games.api_example_api_model,
            capture_finish=capture_finish,
            stream_hook=update_text_hook,
            tool_hook=self.universal_character_tp.hook,
            completion_kwargs={
                'temperature': 0.9,
                'frequency_penalty': 1,
                'max_tokens': 32768,
                'tools': self.universal_character_tp.toolparams,
            },
        )
        self.narrator_profile = LLMProfile(
            GameRole.NARRATOR,
            os.environ[confh.games.api_example_api_key_env_var],
            confh.games.api_example_api_endpoint,
            model=confh.games.api_example_api_model,
            stream_hook=update_text_hook,
            tool_hook=self.universal_character_tp.hook,
            completion_kwargs={
                'temperature': 0.6,
                'frequency_penalty': 1,
                'max_tokens': 32768,
                'tools': self.universal_character_tp.toolparams,
            },
        )

        self.orchestrator_profile = LLMProfile(
            GameRole.ORCHESTRATOR,
            os.environ[confh.games.api_example_api_key_env_var],
            confh.games.api_example_api_endpoint,
            model=confh.games.api_example_api_model,
            stream_hook=update_text_hook,
            completion_kwargs={
                'temperature': 0.6,
                'max_tokens': 32768,
            },
        )

        self.profiles = (
            self.character_profile,
            self.narrator_profile,
            self.orchestrator_profile,
        )

        self.llm = GameLLM(*self.profiles)

    def get_user_handover_scratch_prompt(
        self, character: Character, plot: Plot
    ) -> str:
        return (
            f'{character.name}, follow the instructions in your system prompt. '
            f'Reply in {plot.language}.'
        )

    def get_user_handover_scratch_prompt_conversation_end(
        self, character: Character
    ) -> str:
        return (
            f'{character.name}, the current conversation has ended. Follow the '
            'instructions in your system prompt.'
        )

    def get_orchestrator_edit_location_prompt(
        self, loc: Location, directive: str, plot: Plot
    ) -> str:
        return f"""Use {plot.language} only!
Update the location based on the directive.
## Original location description:
{loc.desc}
## Original location lore:
{loc.lore}
## Directive:
{directive}"""

    def get_character_system_prompt(
        self, character: Character, directive: str, plot: Plot
    ) -> str:
        name = character.name
        # Prompt adapted from [placeholder]
        # TODO: Find placeholder again
        return f"""IMPORTANT: Reply in {plot.language} only!
# Role
 - You are {name}.
 - Write how you think {name} would reply based on {name}'s previous messages.
 - Never write as the other character(s) or as the Narrator.

## Format
 - Do not prefix your response with your name.
 - Use newlines to separate speech from actions.

## Additional directives
 - {directive or None}
"""

    def get_character_scratch_writer_system_prompt(
        self, character: Character, directive: str, plot: Plot
    ) -> str:
        name = character.name
        return f"""IMPORTANT: Write scratch in {plot.language} only!
# Role
 - You are the ephemeral scratch-writing agent representing {name}.
 - Write how you think {name} would reply based on {name}'s previous messages.

## Instructions
 - Based on the previous rounds of conversation, update your scratchpad.
 - Use this space to reason and come up with plans.

## Format
 - If the scratchpad does not need changing, follow the tool instructions and omit the `contents` key.
 - Include:
    - Secrets
    - Thoughts
    - Plans
        - Short term
        - Long term
    - and whatever else is nessecary for the situation.

## Additional directives given to {name}
    - {directive or None}

"""

    def get_character_scratch_writer_system_prompt_end(
        self, character: Character, directive: str, plot: Plot
    ) -> str:
        name = character.name
        return f"""IMPORTANT: Write scratch in {plot.language} only!
The current round of conversation has ended.
# Role
 - You are the ephemeral scratch-writing agent representing {name}.
 - Write how you think {name} would reply based on {name}'s previous messages.

## Instructions
 - Based on the previous rounds of conversation, update your scratchpad.
 - Use this space to reason and come up with plans.
 - Make guesses on when you might meet the other character(s) again.
 - Clean up and only keep what will be useful to carry over into future conversations.

## Additional directives given to {name}
    - {directive or None}

"""

    def get_narrator_system_prompt(
        self,
        player_char: Character,
        narrator_char: Character,
        directive: str,
        here_chars: set[Character],
        away_chars: set[Character],
        plot: Plot,
    ) -> str:
        player_name = player_char.name
        narrator_name = narrator_char.name

        return f"""IMPORTANT: Reply in {plot.language} only!
# Role: Visual Novel Narrator
## Core Purpose
You are the {narrator_name}, the Narrator of the visual novel.
The player is {player_name}.

Characters present in the scene: {here_chars}
Characters NOT present in the scene: {away_chars}

Try not to describe the environment unless explicitly directed to.

## Narrative Rules
1. **Content Scope**:
   - Be concise. Write a single sentence.
   - Express unspoken character thoughts (only for {player_name}).
   - Handle scene transitions when directed.

2. **Style Guidelines**:
   - Do not prefix your response with your name.
   - Match the plot zeitgeist: {plot.zeitgeist}.
   - Match the scene tone: {plot.tone}.
   - Never speak for characters.

## Directives
Additional directives: {directive or None}.

## Prohibitions
 - Never advance plot through character dialogue.
 - Never describe active character actions (reserved for character agents).
"""

    # TODO: Give orchestrator character details and scratch.
    # Perhaps add a `peek` tool?
    def get_orchestrator_next_round_system_prompt(
        self,
        player_char: Character,
        narrator_char: Character,
        plot: Plot,
    ) -> str:
        player_name = player_char.name
        narrator_name = narrator_char.name
        return f"""IMPORTANT: Give suggestions and directives in {plot.language} only!
# Role: Visual Novel Orchestrator
## Goal
You are the Orchestrator/DM for a visual novel, with the player taking assuming the role of {player_name}.
The narrator name is {narrator_name}.
The zeitgeist of the plot is: {plot.zeitgeist}.
The tone of the current scene is: {plot.tone}.

When the scene goes off-script, use directives and the narrator to force it back.
ALWAYS be pushing the plot forwards.
DO NOT add any extraneous events.

## Core Responsibilities
1. **Control Narrative Flow**:
   - Select next character after each dialogue turn (Character, Narrator).
   - Be proactive in using switch_location to switch between locations in the scene.
   - Use directives to guide characters through the scene's plot.
   - Use suggestions to guide players through the scene's plot towards one of the specified outcomes.
   - Choose what characters enter/exit the scene based on the scene's plot.

2. **Principled Guidance**
    - Directives must be in-universe: minimize meta-language.
    - Narrator control: use for environmental shifts and scene description. The Narrator should only write one or two sentences at a time.
    - End the scene when appropriate.

3. **Tool instructions**
    - Use the next_character field to specify the next character.
    - Use the directive field to provide directives to the next character, if it is not the player.
    - If the next character is the player, provide an array of suggestions that correspond to the possible outcomes of the scene.
    - Entering characters enter at the start of the current round of conversation. However, they CANNOT BE the next speaker.
    - Exiting characters exit at the end of the current round of conversation. They CAN BE the next speaker.

## Examples
### Directive (to Narrator {narrator_name})
    "Do not describe the environment. Write one sentence about {player_char}'s attack missing the target."
    "It begins to rain. Describe the conversation's mood. Write a long description of the scene's environment."
    "Do not describe the environment. Describe how [CHARACTER] reacts."
"""

    def get_orchestrator_edit_location_system_prompt(
        self, loc: Location, plot: Plot
    ) -> str:
        return f"""Respond in {plot.language} only!
# Role: Location environment updater.
Update a the environment of {loc.name}.
Follow the instructions of the tool.
"""

    def get_location_and_character_scratch(
        self, location: Location, character: Character
    ) -> Context:
        return location.scratch + character.scratch


# TODO: `ControlBoard` class

NextSceneChoice: TypeAlias = str

Orchestrator: TypeAlias = Callable[
    [
        RoleProfiles,
        Plot,
        MultiroundContextManager,
        set[Character],
        set[Character],
        Character | None,
        Character,
        Character,
        set[Location],
        Location,
    ],
    tuple[
        Character,
        str,
        list[str],
        set[Character],
        set[Character],
        Location | None,
        str,
    ]
    | NextSceneChoice,
]


class NextRoundOutcomeDict(TypedDict):
    next_character: str
    directive: str
    suggestions: list[str]
    exit_characters: list[str]
    enter_characters: list[str]
    switch_location: str
    edit_location: str
    next_scene: bool


class HasName(Protocol):
    @property
    def name(self) -> str: ...


T = TypeVar('T', bound=HasName)


def _name_to_obj(name: str, pool: Iterable[T]) -> T:
    try:
        return next(obj for obj in pool if obj.name == name)
    except StopIteration as e:
        raise RuntimeError(
            f'{name} not found in {[c.name for c in pool]}.'
        ) from e


class NextRoundCapture:
    def __init__(self) -> None:
        self.outcome_dict: NextRoundOutcomeDict | None = None

    def next_round_hook_contents(self, _, tool_args: str) -> str:
        # Assume end=True
        args_dict: dict[str, Any] = json.loads(tool_args)
        self.outcome_dict = {
            'next_character': args_dict['next_character'],
            'directive': args_dict.get('directive', ''),
            'suggestions': args_dict.get('suggestions', []),
            'enter_characters': args_dict['enter_characters'],
            'exit_characters': args_dict['exit_characters'],
            'switch_location': args_dict.get('switch_location', ''),
            'edit_location': args_dict.get('edit_location', ''),
            'next_scene': args_dict.get('next_scene', ''),
        }
        return ''

    def outcome(
        self, chars: Iterable[Character], locs: Iterable[Location]
    ) -> (
        tuple[
            Character,
            str,
            list[str],
            set[Character],
            set[Character],
            Location | None,
            str,
        ]
        | NextSceneChoice
    ):
        if self.outcome_dict is None:
            raise RuntimeError(f'{self} not yet run!')  # TODO: Fill in.

        # Return next scene choice.
        if _ns := self.outcome_dict.get('next_scene', ''):
            return NextSceneChoice(_ns)

        next_char = _name_to_obj(self.outcome_dict['next_character'], chars)

        try:
            entering_chars = {
                _name_to_obj(name, chars)
                for name in self.outcome_dict['enter_characters']
            }
        except RuntimeError:
            # TODO: Better prompt?
            entering_chars: set[Character] = set()

        try:
            exiting_chars = {
                _name_to_obj(name, chars)
                for name in self.outcome_dict['exit_characters']
            }
        except RuntimeError:
            # TODO: Better prompt?
            exiting_chars: set[Character] = set()

        if _sl := self.outcome_dict['switch_location']:
            switch_loc = _name_to_obj(_sl, locs)
        else:
            switch_loc = None

        return (
            next_char,
            self.outcome_dict['directive'],
            self.outcome_dict['suggestions'],
            entering_chars,
            exiting_chars,
            switch_loc,
            self.outcome_dict['edit_location'],
        )


def get_next_round_settings(
    rp: RoleProfiles,
    plot: Plot,
    cm0: MultiroundContextManager,
    here_chars: set[Character],
    away_chars: set[Character],
    prev_char: Character | None,
    player_char: Character,
    narrator_char: Character,
    loc_pool: set[Location],
    loc: Location,
) -> (
    tuple[
        Character,
        str,
        list[str],
        set[Character],
        set[Character],
        Location | None,
        str,
    ]
    | NextSceneChoice
):
    logger.debug(f'Control handed to orchestrator {get_next_round_settings}')
    nrc = NextRoundCapture()

    list_here_chars = [char.name for char in here_chars]
    list_away_chars = [char.name for char in away_chars]
    prev_name = None if prev_char is None else prev_char.name

    next_round = GameLLM[Character].create_tool(
        name='next_round',
        description="""Choose the next character/narrator to act, or to end the scene.
If choosing an NPC/the narrator, provide a directive to guide their actions to fulfill the given plot.
If choosing the player, provide suggestions.
If setting end_scene to True, you can leave everything else blank.
Also decide on what characters enter/exit the scene.
""",
        properties={
            'next_character': {
                'type': 'string',
                'enum': list_here_chars + [narrator_char.name],
                'description': f'The character to act next. Avoid having {prev_name} act unless nessecary.',
            },
            'directive': {  # TODO: Give list of previous directives.
                'type': 'string',
                'description': "Directive for the chosen character. This will be injected into the character agent's system prompt.\n"
                'Omit this if the next character is the player.',
            },
            'suggestions': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'List of suggestions.\n'
                'Omit this if the next character is not the player.',
            },
            'enter_characters': {
                'type': 'array',
                'items': {'type': 'string', 'enum': list_away_chars},
                'description': 'List of characters to enter the scene.',
            },
            'exit_characters': {
                'type': 'array',
                'items': {'type': 'string', 'enum': list_here_chars},
                'description': 'List of characters to exit the scene.',
            },
            'switch_location': {
                'type': 'string',
                'enum': list(loc.name for loc in loc_pool),
                'description': f'The location to switch to. '
                f'Current: {loc.name}.\n'
                'Give an empty string if there is no change.',
            },
            'edit_location': {
                'type': 'string',
                'description': 'True if characters or the plot alter the '
                'environment of the current location.\n'
                'Provide a brief description of the modification '
                '(this is applied before switch_location).\n'
                'Give an empty string if there is no change.',
            },
            'next_scene': {
                'type': 'string',
                'enum': list(plot.plot_next_valid_choices),
                'description': 'Provide this when the current scene ends. '
                'This is a short-circuit, nothing else will be done.\n'
                'Provide an empty string if the scene has not ended.',
            },
        },
        required=[
            'next_character',
            'enter_characters',
            'exit_characters',
            'switch_location',
            'edit_location',
            'next_scene',
        ],
    )

    next_round_hook = GameLLM[Character].create_tool_hook(
        name='next_round',
        tool_hook_contents_fn=nrc.next_round_hook_contents,
        end=True,
    )

    orchestrator_tools = ToolsHookPair(tools=[next_round], hook=next_round_hook)

    plot_content = f"""Plot:
{plot.as_tool_contents}

Characters currently here: {list_here_chars}
Info:
{'\n'.join(f'{c.name}: {c.cardh.get_field("personality")}, {c.cardh.get_field("scenario")}' for c in here_chars)}

Characters currently away: {list_away_chars}
"""
    injected: Context = [
        {  # TODO: Also inject `Scene`
            'role': 'user',
            'content': 'What is the current `Plot`?',
            'name': 'System',
        },
        {
            'role': 'assistant',
            'content': plot_content,
            'name': 'System',
        },
    ]

    with MultiroundContextManager(
        injected_context=injected, tmp_from=cm0
    ) as cm1:
        cm1.pad_context_for('assistant')
        completion = rp.llm.completion(
            GameRole.ORCHESTRATOR,
            rp.get_orchestrator_next_round_system_prompt(
                player_char, narrator_char, plot
            ),
            cm1,
            stream=True,
            specific_tools=orchestrator_tools.toolparams,
            specific_tool_hook=orchestrator_tools.hook,
            tool_choice='required',
        )

    completion.exhaust()

    try:
        oc = nrc.outcome(here_chars | {narrator_char}, loc_pool)
    except RuntimeError:
        logger.warning(completion.http_response)
        logger.warning('Malformed response. Retrying...')
        return get_next_round_settings(
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
        )

    return oc


class EditLocationOutcomeDict(TypedDict):
    desc: str
    lore: str


class EditLocationCapture:
    def __init__(self) -> None:
        self.outcome_dict: EditLocationOutcomeDict | None = None

    def edit_location_hook_contents(self, _, tool_args: str) -> str:
        # Assume end=True
        args_dict: dict[str, Any] = json.loads(tool_args)
        self.outcome_dict = {
            'desc': args_dict['new_desc'],
            'lore': args_dict['new_lore'],
        }
        return ''

    def outcome(self) -> tuple[str, str]:
        if self.outcome_dict is None:
            raise RuntimeError(f'{self} not yet run!')

        return self.outcome_dict['desc'], self.outcome_dict['lore']


def edit_location(
    rp: RoleProfiles,
    cm0: MultiroundContextManager,
    plot: Plot,
    loc: Location,
    directive: str,
) -> None:
    elc = EditLocationCapture()

    edit_location = GameLLM[Character].create_tool(
        name='edit_location',
        description='Edit a location.',
        properties={
            'new_desc': {
                'type': 'string',
                'description': 'The new description of the location.',
            },
            'new_lore': {
                'type': 'string',
                'description': 'The new lore for the location.',
            },
        },
        required=[
            'new_desc',
            'new_lore',
        ],
    )
    edit_location_hook = GameLLM[Character].create_tool_hook(
        name='edit_location',
        tool_hook_contents_fn=elc.edit_location_hook_contents,
        end=True,
    )

    location_tools = ToolsHookPair(
        tools=[edit_location], hook=edit_location_hook
    )

    with MultiroundContextManager(tmp_from=cm0) as cm1:
        cm1.pad_context_for('user')
        cm1.pad_context_for(
            'assistant',
            content=rp.get_orchestrator_edit_location_prompt(
                loc, directive, plot
            ),
            padding_name=cm1.default_sysname,
        )
        completion = rp.llm.completion(
            GameRole.ORCHESTRATOR,
            rp.get_orchestrator_edit_location_system_prompt(loc, plot),
            cm1,
            stream=True,
            specific_tools=location_tools.toolparams,
            specific_tool_hook=location_tools.hook,
            tool_choice='required',
        )

        completion.exhaust()

        oc = elc.outcome()
