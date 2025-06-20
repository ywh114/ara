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
from typing import Any, Callable, Iterable, TypeAlias, TypedDict

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
from world.plot import PlotMarcher

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
            completion_kwargs={'tools': self.universal_character_tp.toolparams},
        )
        self.narrator_profile = LLMProfile(
            GameRole.NARRATOR,
            os.environ[confh.games.api_example_api_key_env_var],
            confh.games.api_example_api_endpoint,
            model=confh.games.api_example_api_model,
            stream_hook=update_text_hook,
            tool_hook=self.universal_character_tp.hook,
            completion_kwargs={'tools': self.universal_character_tp.toolparams},
        )

        self.orchestrator_profile = LLMProfile(
            GameRole.ORCHESTRATOR,
            os.environ[confh.games.api_example_api_key_env_var],
            confh.games.api_example_api_endpoint,
            model=confh.games.api_example_api_model,
            stream_hook=update_text_hook,
        )

        self.profiles = (
            self.character_profile,
            self.narrator_profile,
            self.orchestrator_profile,
        )

        self.llm = GameLLM(*self.profiles)

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
        self, character: Character, directive: str
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
        self, character: Character, directive: str
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
        self, character: Character, directive: str
    ) -> str:
        name = character.name
        return f"""The current round of conversation has ended.
You are the ephemeral scratch-writing agent representing {name}.
Based on the previous rounds of conversation, update your scratchpad.

Make guesses on when you might meet the other character(s) again.

Additional directives: {directive or None}

Clean up and only keep what will be useful to carry over into future conversations.
"""

    def get_narrator_system_prompt(
        self,
        player_char: Character,
        narrator_char: Character,
        directive: str,
        plot: PlotMarcher,
        here_chars: set[Character],
        away_chars: set[Character],
    ) -> str:
        player_name = player_char.name
        narrator_name = narrator_char.name

        return f"""# Role: Visual Novel Narrator
## Core Purpose
You are the {narrator_name}, the Narrator of the story.
The player is {player_name}.
DO NOT prefix with `<{narrator_name}>:`, the system will add these automatically.

Characters present in the scene: {here_chars}
Characters NOT present in the scene: {away_chars}

## Narrative Rules
1. **Content Scope**:
   - Describe environments, weather, and time transitions.
   - Reveal subtle sensory details (sounds, smells, textures).
   - Express unspoken character thoughts (only for {player_name}).
   - Handle scene transitions when directed.

2. **Style Guidelines**:
   - Show, don't tell.
   - Match the plot zeitgeist: {plot.zeitgeist}.
   - Match the scene tone: {plot.scene_tone}.
   - Never speak for characters (except {player_name}'s internal monologue).

## Directives
Additional directives: {directive or None}.

## Prohibitions
- Never advance plot through character dialogue.
- Never describe active character actions (reserved for character agents).
- No meta-commentary about game mechanics.
"""

    def get_orchestrator_next_round_system_prompt(
        self,
        player_char: Character,
        narrator_char: Character,
        plot: PlotMarcher,
    ) -> str:
        player_name = player_char.name
        narrator_name = narrator_char.name
        return f"""# Role: Visual Novel Orchestrator
## Goal
You are the Orchestrator/DM for a visual novel, with the player taking assuming the role of {player_name}.
The narrator name is {narrator_name}.
The zeitgeist of the plot is: {plot.zeitgeist}.
The tone of the current scene is: {plot.scene_tone}.

## Core Responsibilities
1. **Control Narrative Flow**:
   - Select next character after each dialogue turn (Character, Narrator).
   - Use directives to guide characters through the scene's plot.
   - Use suggestions to guide players through the scene's plot towards one of the specified outcomes.
   - Choose what characters enter/exit the scene based on the scene's plot.
   - Maintain scene pacing (60% dialogue / 40% narration).

2. **Principled Guidance**
    - Directives must be in-universe: minimize meta-language.
    - Player freedom: Suggestions are not commands.
    - Narrator control: use for environmental shifts and scene description.
    - End the scene by setting end_scene to True when appropriate.

3. **Tool instructions**
    - Use the next_character field to specify the next character.
    - Use the directive field to provide directives to the next character, if it is not the player.
    - If the next character is the player, provide an array of suggestions that correspond to the possible outcomes of the scene.
    - Entering characters enter at the start of the current round of conversation. However, they CANNOT BE the next speaker.
    - Exiting characters exit at the end of the current round of conversation. They CAN BE the next speaker.


## Examples
### Directive (to Character)
"You decide to confront the suspicious man."
"You continue the conversation."
"You must make a choice: continue to follow {player_name}, or split up."

### Directive (to Narrator {narrator_name})
"{player_char}'s attack misses. Describe the scene."
"It begins to rain. Describe the conversation's mood. Write a long description of the scene's environment."
"Describe how [CHARACTER] reacts."

### Suggestions (to Player {player_name})
["Ask about the dinner.", "Express disinterest."]
["Go to the meeting room.", "Go to the break room.", "Go to the restroom."]"""


Orchestrator: TypeAlias = Callable[
    [
        RoleProfiles,
        PlotMarcher,
        MultiroundContextManager,
        set[Character],
        set[Character],
        Character,
        Character,
    ],
    tuple[Character, str, list[str], set[Character], set[Character], bool]
    | None,
]


class NextRoundOutcomeDict(TypedDict):
    next_character: str
    directive: str
    suggestions: list[str]
    exit_characters: list[str]
    enter_characters: list[str]
    edit_scene_environment: bool
    end_scene: bool


def _name_to_char(name: str, chars: Iterable[Character]) -> Character:
    try:
        return next(char for char in chars if char.name == name)
    except StopIteration as e:
        raise RuntimeError(
            f'{name} not found in {[c.name for c in chars]}'
        ) from e  # TODO: Fill in


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
            'edit_scene_environment': args_dict.get(
                'edit_scene_environment', False
            ),
            'end_scene': args_dict['end_scene'],
        }
        return ''

    def outcome(
        self, chars: Iterable[Character]
    ) -> (
        tuple[Character, str, list[str], set[Character], set[Character], bool]
        | None
    ):
        if self.outcome_dict is None:
            raise RuntimeError(f'{self} not yet run!')  # TODO: Fill in.

        # End scene -> None
        if self.outcome_dict['end_scene']:
            return None

        next_char = _name_to_char(self.outcome_dict['next_character'], chars)
        try:
            entering_chars = {
                _name_to_char(name, chars)
                for name in self.outcome_dict['enter_characters']
            }
        except RuntimeError:
            # TODO: Better prompt?
            entering_chars = set()
        try:
            exiting_chars = {
                _name_to_char(name, chars)
                for name in self.outcome_dict['exit_characters']
            }
        except RuntimeError:
            # TODO: Better prompt?
            exiting_chars = set()

        return (
            next_char,
            self.outcome_dict['directive'],
            self.outcome_dict['suggestions'],
            entering_chars,
            exiting_chars,
            self.outcome_dict['edit_scene_environment'],
        )


def get_next_round_settings(
    rp: RoleProfiles,
    plot: PlotMarcher,
    cm0: MultiroundContextManager,
    here_chars: set[Character],
    away_chars: set[Character],
    player_char: Character,
    narrator_char: Character,
) -> (
    tuple[Character, str, list[str], set[Character], set[Character], bool]
    | None
):
    logger.debug(f'Control handed to orchestrator {get_next_round_settings}')
    nrc = NextRoundCapture()

    list_here_chars = [char.name for char in here_chars]
    list_away_chars = [char.name for char in away_chars]

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
                'description': 'The character to act next.',
            },
            'directive': {
                'type': 'string',
                'description': "Directive for the chosen character. This will be injected into the character agent's system prompt.\n"
                'You can leave omit this if the next character is the player.',
            },
            'suggestions': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'List of suggestions. You can leave omit this if the next character is not the player.',
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
            'edit_scene_environment': {
                'type': 'boolean',
                'description': 'True if characters or the plot alter the '
                'environment of the scene. [Currently unimplemented]',
            },
            'end_scene': {
                'type': 'boolean',
                'description': 'True if the scene ends now, false otherwise.',
            },
        },
        required=[
            'next_character',
            'enter_characters',
            'exit_characters',
            'end_scene',
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
        rp.llm.completion(
            GameRole.ORCHESTRATOR,
            rp.get_orchestrator_next_round_system_prompt(
                player_char, narrator_char, plot
            ),
            cm1,
            stream=True,
            specific_tools=orchestrator_tools.toolparams,
            specific_tool_hook=orchestrator_tools.hook,
            tool_choice='required',
        ).exhaust()

    oc = nrc.outcome(here_chars | {narrator_char})

    return oc
