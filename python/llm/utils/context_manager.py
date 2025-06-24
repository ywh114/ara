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
import operator
import re
from functools import reduce
from pprint import pp
from typing import (
    ContextManager,
    Generic,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    override,
)

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as Function1,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as Function0,
)
from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

Context: TypeAlias = list[ChatCompletionMessageParam]
"""
Type alias for conversation context as a list of chat completion messages.
"""


class MultiroundContextManager(ContextManager, Generic[T]):
    """
    Context manager for multi-round conversations with multiple entities.

    Manages conversation state including:
    - Injected base context.
    - Active conversation history.
    - Entity presence tracking.
    - Message sequencing constraints.

    :param entities: Initial entities in the conversation.
    :param injected_context: Pre-existing context to prepend to all
    conversations.
    :param tmp_from: Base context manager to clone for temporary contexts.
    """

    default_sysname = 'System'
    """Default name for system messages."""

    def __init__(
        self,
        *entities: T,
        injected_context: Context | None = None,
        tmp_from: Self | None = None,
    ) -> None:
        self.base = tmp_from is None
        if self.base:
            assert tmp_from is None
            self.injected_context: Context = injected_context or []
            self.context: Context = []
            self.head: ChatCompletionMessageParam | None = None
            self.seen_entities: dict[T, list[slice]] = {
                entity: [] for entity in entities
            }

            self.present_entities: set[T] = set()
        else:
            assert tmp_from is not None
            # Create a temporary context.
            self.injected_context: Context = (
                injected_context or tmp_from.injected_context.copy()
            )
            self.context: Context = tmp_from.context.copy()
            self.head: ChatCompletionMessageParam | None = (
                tmp_from.head.copy() if tmp_from.head else None
            )
            self.present_entities: set[T] = tmp_from.present_entities.copy()
            self.seen_entities: dict[T, list[slice]] = (
                tmp_from.seen_entities.copy()
            )

    def context_of(self, entity: T) -> Context:
        """
        Get the conversation context visible to a specific entity.

        :param entity: The target entity.
        :return: Conversation context visible to the entity.
        :raises RuntimeError: If entity is not being tracked.
        """
        try:
            return reduce(
                operator.add,
                (self.context[sl] for sl in self.seen_entities[entity]),
            )
        except KeyError as e:
            raise RuntimeError(
                f'{entity} not in {set(self.seen_entities.keys())}'
            ) from e

    def filter_to(self, entity: T, confirm_destructive: bool = False) -> None:
        """
        Filter the current context to an entity's view.

        :param entity: Entity whose view to adopt.
        :param confirm_destructive: Must be True for base context operations.
        :raises RuntimeError: For destructive operations without confirmation.
        """
        if not confirm_destructive:
            if self.base:
                raise RuntimeError(
                    'Cannot perform a destructive conversation filter '
                    'unless `confirm_destructive` is `True`.'
                )
        self.context = self.context_of(entity)
        self.head = self.context[-1] if self.context else None

    @override
    def __enter__(self) -> Self:
        """Enter the runtime context."""
        return self

    @override
    def __exit__(self, *_) -> None:
        """Exit the runtime context."""
        # Simpler to leave the exit logic outside of the environment.

    def enter_entities(self, *entities: T) -> None:
        """
        Mark entities as entering the conversation.

        :param entities: Entities entering the conversation.
        :raises RuntimeError: If entity is not being tracked.
        """
        for entity in entities:
            sl = slice(len(self.context), None)

            if entity not in self.seen_entities:
                raise RuntimeError(f'{entity} is not in the scene.')
            else:
                if not (_slices := self.seen_entities[entity]):
                    pass
                else:
                    last_appearance = _slices[-1]
                    if last_appearance.stop is None:
                        logger.warning(
                            f'{entity} is already present, ignoring.'
                        )
                        continue  # TODO: Remove when `enum` enforced by API.
                        raise RuntimeError(
                            f'{entity} has already entered the scene.'
                        )
                self.seen_entities[entity] += [sl]

        self.present_entities |= set(entities)

    def exit_entities(self, *entities: T) -> None:
        """
        Mark entities as exiting the conversation.

        :param entities: Entities exiting the conversation.
        :raises RuntimeError: If entity is not being tracked.
        """
        for entity in entities:
            if entity not in self.seen_entities:
                raise RuntimeError(f'{entity} is not in the scene.')
            last_appearance = self.seen_entities[entity][-1]
            if last_appearance.stop is not None:
                logger.warning(f'{entity} is already off-scene, ignoring.')
                continue  # TODO: Remove when `enum` enforced by API.
                raise RuntimeError(f'{entity} has already exited the scene.')

            sl = slice(last_appearance.start, len(self.context))

            self.seen_entities[entity][-1] = sl

        self.present_entities -= set(entities)

    def user_message(
        self,
        _content: str,
        name=default_sysname,
        suppress_decorations: bool = False,
    ) -> Self:
        """
        Add a user message to the conversation.

        :param _content: Message content.
        :param name: Name of the user. Defaults to the system name.
        :param suppress_decorations: Whether to skip automatic name decorations.
        :return: Self for method chaining.
        :raises RuntimeError: If message violates sequencing rules.
        """
        if self.head is not None and self.head['role'] != 'assistant':
            pp(self.context)
            raise RuntimeError(
                'User message must come after system or assistant message.'
            )

        if suppress_decorations:
            content = _content
        else:
            content = f'<{name}>: ' + (_content or '...')

        # TODO: Find better solution
        content = _content

        self.head = {
            'role': 'user',
            'content': content,
            'name': name,
        }

        self.context.append(self.head)

        return self

    def tool_message(self, content: str, tool_call_id: str) -> Self:
        """
        Add a tool result message to the conversation.

        :param content: Tool execution result.
        :param tool_call_id: ID of the tool call being responded to.
        :return: Self for method chaining.
        :raises RuntimeError: If message violates sequencing rules.
        """
        if (
            self.head is None
            or self.head['role'] != 'assistant'
            or ('tool_calls' in self.head and not self.head['tool_calls'])
        ):
            pp(self.context)
            raise RuntimeError('Tool call must come after assistant tool call.')
        self.head = {
            'role': 'tool',
            'content': content,
            'tool_call_id': tool_call_id,
        }

        self.context.append(self.head)

        return self

    def assistant_message(
        self,
        _content: str,
        tool_calls: list[ChatCompletionMessageToolCall],
        name: str = default_sysname,
        suppress_decorations: bool = False,
    ) -> Self:
        """
        Add an assistant message to the conversation.

        :param _content: Message content.
        :param tool_calls: List of tool calls included in the message. Leave as
        `[]` if none.
        :param name: Name of the user. Defaults to the system name.
        :param suppress_decorations: Whether to skip automatic name decorations.
        :return: Self for method chaining.
        :raises RuntimeError: If message violates sequencing rules.
        """
        if self.head is None or self.head['role'] == 'assistant':
            pp(self.context)
            raise RuntimeError(
                'Assistant message must come after user message or tool call.'
            )
        tc: list[ChatCompletionMessageToolCallParam] = [
            ChatCompletionMessageToolCallParam(
                id=tool.id,
                function=Function0(
                    arguments=tool.function.arguments, name=tool.function.name
                ),  # XXX: `Function` needs to be recast.
                type=tool.type,
            )
            for tool in tool_calls
        ]
        if suppress_decorations:
            content = _content
        else:
            decor = f'<{name}>:'
            content = (
                f'{decor} ' + (re.sub(rf'^({decor})*', '', _content) or '...')
            ).strip()  # Attempt to dedupe decorations.

        # TODO: Find better solution
        content = _content

        self.head = {
            'role': 'assistant',
            'content': content,
            'tool_calls': tc,
            'name': name,
        }
        # Discard `tool_calls` key if no tool calls.
        if not tc:
            self.head.pop('tool_calls')  # No `del` for `TypedDict`

        self.context.append(self.head)

        return self

    def concat_context(self, context: Context) -> Self:
        """
        Append an existing context to the current conversation.

        :param context: Context to append.
        :return: Self for method chaining.
        """
        for line in context.copy():
            content = line.get('content', '')

            assert isinstance(content, str)

            if line['role'] == 'user':
                self.user_message(
                    content,
                    name=line.get('name', self.default_sysname),
                )
            elif line['role'] == 'assistant':
                # FIXME: Rewrite this or `self.assistant_message` to skip
                # extra conversions
                tool_calls: list[ChatCompletionMessageToolCall] = [
                    ChatCompletionMessageToolCall(
                        id=tool['id'],
                        function=Function1(
                            arguments=tool['function']['arguments'],
                            name=tool['function']['name'],
                        ),  # XXX: `Function` needs to be recast.
                        type=tool['type'],
                    )
                    for tool in line.get('tool_calls', [])
                ]
                self.assistant_message(
                    content,
                    tool_calls=tool_calls,
                    name=line.get('name', self.default_sysname),
                )
            elif line['role'] == 'tool':
                self.tool_message(content, tool_call_id=line['tool_call_id'])

        return self

    def pad_context_for(
        self,
        who_is_next: Literal['user', 'assistant', 'scratch'],
        *,
        content: str = '',
        padding_name: str = 'Padding',
        base: bool = False,
    ) -> None:
        """
        Add padding messages to satisfy conversation sequence rules.

        :param who_is_next: Next expected message type.
        :param content: Content for padding messages.
        :param padding_name: Name to use for padding messages.
        :param base: Must be `True` when modifying base context.
        :raises RuntimeError: If padding violates context rules.
        """
        if self.base and not base:
            raise RuntimeError(
                'Cannot pad base context without setting '
                '`allow_pad_base` to `True`.'
            )
        head = 'system' if self.head is None else self.head['role']
        if who_is_next in ('user', 'scratch') and head == 'user':
            self.assistant_message(
                content,
                tool_calls=[],
                name=padding_name,
                suppress_decorations=True,
            )
        elif who_is_next == 'assistant' and head != 'user':
            self.user_message(
                content, name=padding_name, suppress_decorations=True
            )

    def to_list(self, entity: T | None = None) -> Context:
        """
        Get full conversation context as a list.

        :param entity: If provided, returns context visible to this entity.
        :return: Complete conversation context.
        """
        return self.injected_context + (
            self.context_of(entity) if entity else self.context
        )
