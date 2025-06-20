#!/usr/bin/env python3


from typing import Iterable
from world.character_roles import Plot, RoleProfiles
from world.conversation import multiround_conversation

from examples.character import char0, char1, player, narrator
from examples.config import confh

rp = RoleProfiles(confh)

plot = Plot(
    character_pool={char0, char1, player},
    starting_characters={char0, char1, player},
)


def user_prompt_from_input(suggestions: Iterable[str]) -> str:
    print('\n'.join(suggestions))
    return input('User> ')


multiround_conversation(
    rp,
    plot,
    player_char=player,
    narrator_char=narrator,
    get_user_prompt=user_prompt_from_input,
)
