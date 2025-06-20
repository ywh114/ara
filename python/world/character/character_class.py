#!/usr/bin/env python3
# TODO: Add Character to __init__.py
from dataclasses import dataclass
from typing import Any, Self
from uuid import UUID

from world.character.card import CardHolder
from world.character.memory import Memory
from llm.utils.context_manager import Context
from utils.bars import BarManager
from world.importance import ImportanceEnum


@dataclass
class Character:
    id: UUID
    cardh: CardHolder
    memory: Memory[Self]
    bars: BarManager
    capabilities: list[str]  # "tools" that do not result in tool calls.
    importance: ImportanceEnum

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
                'content': '<System>: Please provide your `name`.',
                'name': 'System',
            },
            {
                'role': 'assistant',
                'content': f'<{name}>: {name}',
                'name': name,
            },
            {
                'role': 'user',
                'content': '<System>: Please provide your `summary`.',
                'name': 'System',
            },
            {
                'role': 'assistant',
                'content': f'<{name}>: {self.cardh.get_field("summary")}',
                'name': name,
            },
            {
                'role': 'user',
                'content': '<System>: Please provide your `personality`.',
                'name': 'System',
            },
            {
                'role': 'assistant',
                'content': f'<{name}>: {self.cardh.get_field("personality")}',
                'name': name,
            },
            {
                'role': 'user',
                'content': '<System>: Please provide your `scenario`.',
                'name': 'System',
            },
            {
                'role': 'assistant',
                'content': f'<{name}>: {self.cardh.get_field("scenario")}',
                'name': name,
            },
            {
                'role': 'user',
                'content': '<System>: Please provide your `greeting_message`.',
                'name': 'System',
            },
            {
                'role': 'assistant',
                'content': f'<{name}>: {self.cardh.get_field("greeting_message")}',
                'name': name,
            },
            {
                'role': 'user',
                'content': '<System>: Please provide your `example_messages`.',
                'name': 'System',
            },
            {
                'role': 'assistant',
                'content': f'<{name}>: {self.cardh.get_field("example_messages")}',
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
                'name': 'System',
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
