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
# TODO: Refactor tools for Orchestrator,
# write `ControlPanel` class instead.
from dataclasses import dataclass, field
from typing import TypeVar

from configuration.config import Config
from configuration.settings import DefaultSettings, Settings

A = TypeVar('A')


def dfac(obj: A) -> A:
    return field(default_factory=lambda: obj)


@dataclass
class PlotSettings(Settings):
    # TODO: Fill out :ivar:
    """
    Global application settings.

    :ivar:
    """

    id: str
    language: str
    zeitgeist: str
    tone: str
    # [character]
    character_pool: list[str]
    character_inits: list[str]
    character_player: str
    character_narrator: str
    # [location]
    location_pool: list[str]
    location_init: str
    location_time: str
    location_rels: str
    location_descs: dict[str, str]
    location_deltas: dict[str, str]
    # [plot]
    plot_considerations: str
    plot_story: str
    plot_next: dict[str, dict[str, str | list[str]]]


@dataclass
class DefaultPlotSettings(DefaultSettings[PlotSettings]):
    """Default plot settings."""

    id: str = ''
    language: str = ''
    zeitgeist: str = ''
    tone: str = ''
    # [character]
    character_pool: list[str] = dfac([])
    character_inits: list[str] = dfac([])
    character_player: str = ''
    character_narrator: str = ''
    # [location]
    location_pool: list[str] = dfac([])
    location_init: str = ''
    location_time: str = ''
    location_rels: str = ''
    location_descs: dict[str, str] = dfac({})
    location_deltas: dict[str, str] = dfac({})
    # [plot]
    plot_considerations: str = ''
    plot_story: str = ''
    plot_next: dict[str, dict[str, str | list[str]]] = dfac({})


@dataclass
class PlotConfig(Config[PlotSettings, DefaultPlotSettings]):
    """Game-specific configuration container."""

    settings: 'PlotSettings'
