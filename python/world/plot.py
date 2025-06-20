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
from world.character.character_class import Character


class PlotMarcher:  # FIXME: Implement.
    def __init__(
        self,
        character_pool: set[Character],
        starting_characters: set[Character],
    ) -> None:
        self._character_pool = character_pool
        self._starting_characters = starting_characters

    @property
    def character_pool(self) -> set[Character]:
        # TODO:
        return self._character_pool

    @property
    def starting_characters(self) -> set[Character]:
        # TODO:
        return self._starting_characters

    @property
    def scene_outcomes(self) -> list[str]:
        # TODO:
        return ['debug 0', 'debug 1']

    @property
    def scene_outcomes_pretty(self) -> str:
        return '\n'.join(f'{i}, {s}' for i, s in enumerate(self.scene_outcomes))

    @property
    def scene_tone(self) -> str:
        # TODO:
        return 'debug'

    @property
    def zeitgeist(self) -> str:
        # TODO:
        return 'debug'

    @property
    def as_tool_contents(self) -> str:
        # TODO:
        return (
            'I am debugging. Cycle between characters/narrator and only exit when the user (me, doing the debugging) mentions exiting to you.\n'
            'Debug plot: the characters are conversing. Make sure everyone has a chance to speak. Player goes first.\n'
        )
