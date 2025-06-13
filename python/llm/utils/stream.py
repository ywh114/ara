#!/usr/bin/env python3
from abc import ABC
from dataclasses import dataclass
from functools import update_wrapper
from typing import (
    Callable,
    Generator,
    Generic,
    Iterable,
    Iterator,
    Self,
    TypeAlias,
    TypeVar,
    reveal_type,
)
from itertools import chain

from openai import Stream

T = TypeVar('T')


class ReasoningContentStr(str):
    """A string subclass that represents reasoning content."""


class ContentStr(str):
    """A string subclass that represents regular content."""


DeltaStr: TypeAlias = ReasoningContentStr | ContentStr
"""
Type alias for string deltas in streamed responses.

Can represent either:
- :class:`ReasoningContentStr` (reasoning segments)
- :class:`ContentStr` (regular content segments)
"""


class CustomResponseStr(str):
    """
    A string subclass with additional attributes for reasoning and content.

    :ivar reasoning_content: A string representing reasoning content.
    :ivar content: A string representing regular content.
    """

    reasoning_content: ReasoningContentStr
    content: ContentStr

    def __new__(
        cls, value: str = '', reasoning_content: str = '', content: str = ''
    ) -> Self:
        """Create a new ResponseStr instance.

        :param value: The base string value.
        :param reasoning_content: Initial reasoning content.
        :param content: Initial regular content.
        :return: A new instance with the provided values.
        """
        instance = super().__new__(cls, value)
        instance.reasoning_content = ReasoningContentStr(reasoning_content)
        instance.content = ContentStr(content)
        return instance

    def __add__(self, other: str | DeltaStr | Self) -> Self:
        """
        Addition operation for `ResponseStr`.
        Treat other all other subclasses of `str` as `ContentStr`.

        :param other: The value to add.
        :return: New instance with concatenated values.
        """
        new_str = super().__add__(other)
        updated = self.__class__(new_str, self.reasoning_content, self.content)

        if isinstance(other, self.__class__):
            updated.reasoning_content = ReasoningContentStr(
                updated.reasoning_content + other.reasoning_content
            )
            updated.content = ContentStr(updated.content + other.content)
        elif isinstance(other, ReasoningContentStr):
            updated.reasoning_content = ReasoningContentStr(
                updated.reasoning_content + other
            )
        elif isinstance(other, (ContentStr, str)):
            updated.content = ContentStr(updated.content + other)

        return updated

    def __iadd__(self, other: DeltaStr | Self) -> Self:
        """
        In-place addition for `ResponseStr`.
        Treat other all other subclasses of `str` as `ContentStr`.

        :param other: The value to add.
        :return: Updated instance with concatenated values.
        """
        return self.__add__(other)


CompletionFn: TypeAlias = Callable[[str], CustomResponseStr | Generator[T]]


@dataclass
class CustomStreamHookArgs(Generic[T]):
    """
    Container for stream processing arguments.

    Attributes:
        called_by: The function this was called by.
        head: Most recent chunk from stream.
        chunks: All accumulated chunks.
        head_as_str: String representation of head chunk.
        text: Aggregated response content.
    """

    called_by: CompletionFn[T]
    head: T
    chunks: list[T]
    head_as_str: DeltaStr
    text: CustomResponseStr


CustomChunkToStrFnType: TypeAlias = Callable[[T], DeltaStr]
"""
Type alias for chunk conversion functions.

Signature:
    ``(chunk: T) -> DeltaStr``
"""

CustomStreamHookFnType: TypeAlias = Callable[[CustomStreamHookArgs[T]], None]
"""
Type alias for stream hook functions.

Signature:
    ``(args: CustomStreamHookArgs[T]) -> None``
"""


class CustomChunkToStr(Generic[T]):
    """
    Decorator for chunk-to-string conversion functions.

    Usage:
        @CustomChunkToStr
        def my_converter(chunk: T) -> DeltaStr: ...
    """

    def __init__(self, fn: CustomChunkToStrFnType[T]) -> None:
        self.to_str = fn
        update_wrapper(self, fn)

    def __call__(self, chunk: T) -> DeltaStr:
        return self.to_str(chunk)


class CustomStreamHook(Generic[T]):
    def __init__(self, *funcs: CustomStreamHookFnType[T]) -> None:
        self.hooks = funcs

    def __call__(
        self, args: CustomStreamHookArgs[T]
    ) -> Iterable[Generator[T] | None]:
        return (hook(args) for hook in self.hooks)

    def __or__(self, other: 'CustomStreamHook[T]') -> 'CustomStreamHook[T]':
        return CustomStreamHook(*self.hooks, *other.hooks)


@CustomChunkToStr
def to_str_not_implemented(_) -> DeltaStr:
    """Default chunk converter (returns empty `ContentStr`)."""
    return ContentStr()


@CustomStreamHook
def no_hook(_: CustomStreamHookArgs[T]) -> None:
    """Default no-op stream hook."""


class CustomStreamHandler(Iterator[T]):
    def __init__(
        self,
        stream: Stream[T],
        to_str: CustomChunkToStr[T] = to_str_not_implemented,
        hook: CustomStreamHook[T] = no_hook,
        called_by: CompletionFn[T] = lambda _: CustomResponseStr(),
    ) -> None:
        self.stream = stream
        self.to_str = to_str
        self.hook = hook
        self.called_by = called_by
        self._generator = (chunk for chunk in stream)

        self.head: T | None = None
        self.chunks: list[T] = []

        self.text: CustomResponseStr = CustomResponseStr()

        self.response = self.stream.response

    def __next__(self) -> T:
        chunk = next(self._generator)

        self.head = chunk
        self.chunks += [self.head]
        self.text += (head_as_str := self.to_str(self.head))

        # TODO: Allow hooks to extend (replace) the _generator.
        possible_extensions = self.hook(
            CustomStreamHookArgs(
                self.called_by, self.head, self.chunks, head_as_str, self.text
            )
        )

        for extn in (extn for extn in possible_extensions if extn is not None):
            # Extend the generator.
            self._generator = chain(self._generator, extn)

        return chunk

    def close(self) -> None:
        """Closes the underlying stream."""
        return self.stream.close()

    def exhaust(self) -> CustomResponseStr:
        """Exhaust the iterator."""
        _ = tuple(_ for _ in self)
        return self.text
