#!/usr/bin/env python3
import operator
from functools import reduce, update_wrapper
from pprint import pp
import re
from typing import (
    Callable,
    ContextManager,
    Generic,
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
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as Function0,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as Function1,
)
from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

Context: TypeAlias = list[ChatCompletionMessageParam]
RegisterHookFnType: TypeAlias = Callable[[T, Context], None]


class RegisterHook(Generic[T]):
    def __init__(self, fn: RegisterHookFnType[T]) -> None:
        self.hook = fn
        update_wrapper(self, fn)

    def __call__(self, entity: T, context: Context) -> None:
        return self.hook(entity, context)


@RegisterHook
def no_register_hook(*_) -> None:
    """Default no-op register hook."""


class MultiroundContextManager(ContextManager, Generic[T]):
    default_user_name = 'User'
    default_assistant_name = 'Unknown'

    def __init__(
        self,
        *entities: T,
        injected_context: Context | None = None,
        register_hook: RegisterHook[T] = no_register_hook,
        tmp_from: Self | None = None,
    ) -> None:
        self.base = tmp_from is None
        if self.base:
            assert tmp_from is None
            self.injected_context: Context = injected_context or []
            self.context: Context = []
            self.head: ChatCompletionMessageParam | None = None
            self.seen_entities: dict[T, list[slice]] = {
                entity: [slice(0, None)] for entity in entities
            }

            self.present_entities: set[T] = set()
            # Called on exit.
            self.register_hook: RegisterHook[T] = register_hook
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
            # No register hook for temporary context.
            self.register_hook: RegisterHook[T] = no_register_hook

    def context_of(self, entity: T) -> Context:
        return reduce(
            operator.add,
            (self.context[sl] for sl in self.seen_entities[entity]),
        )

    def filter_to(self, entity: T, confirm_destructive: bool = False) -> None:
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
        return self

    @override
    def __exit__(self, *_) -> None:
        for entity, slices in self.seen_entities.items():
            self.register_hook(
                entity,
                reduce(operator.add, (self.context[sl] for sl in slices)),
            )

    def enter_entities(self, *entities: T) -> None:
        for entity in entities:
            sl = slice(len(self.context), None)

            if entity not in self.seen_entities:
                raise RuntimeError(f'{entity} is not in the scene.')
            else:
                last_appearance = self.seen_entities[entity][-1]
                if last_appearance.stop is None:
                    continue  # TODO: Better prompt.
                    raise RuntimeError(
                        f'{entity} has already entered the scene.'
                    )
                self.seen_entities[entity] += [sl]

        self.present_entities |= set(entities)

    def exit_entities(self, *entities: T) -> None:
        for entity in entities:
            if entity not in self.seen_entities:
                raise RuntimeError(f'{entity} is not in the scene.')
            last_appearance = self.seen_entities[entity][-1]
            if last_appearance.stop is not None:
                continue  # TODO: Better prompt.
                raise RuntimeError(f'{entity} has already exited the scene.')

            sl = slice(last_appearance.start, len(self.context))

            self.seen_entities[entity][-1] = sl

        self.present_entities -= set(entities)

    def user_message(
        self,
        content: str,
        name=default_user_name,
        suppress_decorations: bool = False,
    ) -> Self:
        if self.head is not None and self.head['role'] != 'assistant':
            pp(self.context)
            raise RuntimeError(
                'User message must come after system or assistant message.'
            )

        if suppress_decorations:
            _content = content
        else:
            _content = f'<{name}>: {content}'

        self.head = {
            'role': 'user',
            'content': _content,
            'name': name,
        }

        self.context.append(self.head)

        return self

    def tool_message(self, content: str, tool_call_id: str) -> Self:
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
        content: str,
        tool_calls: list[ChatCompletionMessageToolCall],
        name: str = default_assistant_name,
        suppress_decorations: bool = False,
    ) -> Self:
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
            _content = content
        else:
            _decoration = f'<{name}>:'
            _content = (
                f'{_decoration} {re.sub(rf"^({_decoration})*", "", content)}'
            ).strip()  # Dedupe decorations.
        if tc:
            self.head = {
                'role': 'assistant',
                'content': f'<{name}>: {_content}',
                'tool_calls': tc,
                'name': name,
            }
        else:
            self.head = {'role': 'assistant', 'content': _content}

        self.context.append(self.head)

        return self

    def concat_context(self, context: Context) -> Self:
        for line in context.copy():
            content = line.setdefault('content', '')

            assert isinstance(content, str)

            if line['role'] == 'user':
                self.user_message(
                    content,
                    name=line.setdefault('name', self.default_user_name),
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
                    for tool in line.setdefault('tool_calls', [])
                ]
                self.assistant_message(
                    content,
                    tool_calls=tool_calls,
                    name=line.setdefault('name', self.default_assistant_name),
                )
            elif line['role'] == 'tool':
                self.tool_message(content, tool_call_id=line['tool_call_id'])

        return self

    def tolist(self, entity: T | None = None) -> Context:
        return self.injected_context + (
            self.context_of(entity) if entity else self.context
        )
