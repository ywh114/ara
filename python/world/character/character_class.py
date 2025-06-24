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
# TODO: Add Character to __init__.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, override
from uuid import UUID

from configuration import ConfigHolder, P, Q, V, X
from llm.database import DatabaseProvider
from llm.utils.context_manager import Context, MultiroundContextManager
from utils.uuid4_from_seed import uuid4_from_seed
from world.character.card import CardHolder
from world.character.card import standard_load as standard_load_
from world.character.memory import Memory
from world.importance import CharacterImportance, ImportanceEnum

# from utils.bars import BarManager

_image_sfx = '.png'
_importance_sfx = '.m'


@dataclass
class Character:
    id: UUID
    cardh: CardHolder
    memory: Memory[Self]
    # bars: BarManager
    # capabilities: list[str]  # "tools" that do not result in tool calls.
    importance: ImportanceEnum
    skins: list[str]

    @property
    def name(self) -> str:
        return self.cardh.get_field('name')

    @property
    def whoami(self) -> Context:
        # XXX: To be inserted directly without processing.
        name = self.cardh.get_field('name')
        return [
            {
                'role': 'user',
                'content': 'Please provide your `name`.',
                'name': MultiroundContextManager.default_sysname,
            },
            {
                'role': 'assistant',
                'content': f'{name}',
                'name': name,
            },
            {
                'role': 'user',
                'content': 'Please provide your `summary`.',
                'name': MultiroundContextManager.default_sysname,
            },
            {
                'role': 'assistant',
                'content': f'{self.cardh.get_field("summary")}',
                'name': name,
            },
            {
                'role': 'user',
                'content': 'Please provide your `personality`.',
                'name': MultiroundContextManager.default_sysname,
            },
            {
                'role': 'assistant',
                'content': f'{self.cardh.get_field("personality")}',
                'name': name,
            },
            {
                'role': 'user',
                'content': 'Please provide your `scenario`.',
                'name': MultiroundContextManager.default_sysname,
            },
            {
                'role': 'assistant',
                'content': f'{self.cardh.get_field("scenario")}',
                'name': name,
            },
            {
                'role': 'user',
                'content': 'Please provide your `greeting_message`.',
                'name': MultiroundContextManager.default_sysname,
            },
            {
                'role': 'assistant',
                'content': f'{self.cardh.get_field("greeting_message")}',
                'name': name,
            },
            {
                'role': 'user',
                'content': 'Please provide your `example_messages`.',
                'name': MultiroundContextManager.default_sysname,
            },
            {
                'role': 'assistant',
                'content': f'{self.cardh.get_field("example_messages")}',
                'name': name,
            },
        ]

    @property
    def scratch(self) -> Context:
        """Must come after `assistant` message."""
        empty = self.memory.scratch == self.memory.default_scratch
        if not empty:
            user_content = (
                'Please provide your `scratch`.\n'
                'Never show this to others, or mention that it exists!'
            )
            assistant_content = self.memory.scratch
        else:
            user_content = (
                'Please provide the summary of your `scratch` '
                'from your previous conversation.\n'
                'Never show this to others, or mention that it exists!'
            )
            assistant_content = self.memory.prev_scratch

        return [
            {
                'role': 'user',
                'content': user_content,
                'name': MultiroundContextManager.default_sysname,
            },
            {
                'role': 'assistant',
                'content': assistant_content,
                'name': self.cardh.get_field('name'),
            },
        ]

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: Any) -> bool:
        assert isinstance(value, Character)
        return self.id == value.id

    @override
    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.name}>'


def standard_load(
    name: str, db: DatabaseProvider, confh: ConfigHolder[P, Q, V, X]
) -> Character:
    path = confh.insts.cache_assets_cc_dir.joinpath(name)
    card = Path(name, name).with_suffix(_image_sfx)

    cardh = standard_load_(card, confh)

    try:
        importance = next(
            f.stem for f in path.iterdir() if f.suffix == _importance_sfx
        )
    except StopIteration:
        # TODO: Support ILevel
        importance = 'ANONYMOUS'

    _id = uuid4_from_seed(name + '_char')
    return Character(
        _id,
        cardh,
        Memory(db, key=_id),
        getattr(CharacterImportance, importance),
        [f.stem for f in path.iterdir() if f.suffix == _image_sfx],
    )
