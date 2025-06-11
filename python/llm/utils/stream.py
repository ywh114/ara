#!/usr/bin/env python3
from dataclasses import dataclass
from functools import update_wrapper
from typing import (
    Callable,
    Generic,
    Iterator,
    Self,
    TypeAlias,
    TypeVar,
)

from openai import Stream

T = TypeVar('T')


class ReasoningContentStr(str):
    """A string subclass that represents reasoning content."""


class ContentStr(str):
    """A string subclass that represents regular content."""


DeltaStr: TypeAlias = ReasoningContentStr | ContentStr


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


@dataclass
class CustomStreamHookArgs(Generic[T]):
    head: T
    chunks: list[T]
    head_as_str: DeltaStr
    text: CustomResponseStr


# For readability.
CustomChunkToStrFnType: TypeAlias = Callable[[T], DeltaStr]
CustomStreamHookFnType: TypeAlias = Callable[[CustomStreamHookArgs[T]], None]


class CustomChunkToStr(Generic[T]):
    """A typed decorator for chunk conversion."""

    def __init__(self, fn: CustomChunkToStrFnType[T]) -> None:
        self.to_str = fn
        update_wrapper(self, fn)

    def __call__(self, chunk: T) -> DeltaStr:
        return self.to_str(chunk)


class CustomStreamHook(Generic[T]):
    """A typed decorator for stream hooks."""

    def __init__(self, fn: CustomStreamHookFnType[T]) -> None:
        self.hook = fn
        update_wrapper(self, fn)

    def __call__(self, args: CustomStreamHookArgs[T]) -> None:
        self.hook(args)


@CustomChunkToStr
def to_str_not_implemented(_) -> DeltaStr:
    return ContentStr()


@CustomStreamHook
def no_hook(_: CustomStreamHookArgs[T]) -> None:
    pass


class CustomStreamHandler(Iterator[T]):
    def __init__(
        self,
        stream: Stream[T],
        to_str: CustomChunkToStr[T] = to_str_not_implemented,
        hook: CustomStreamHook[T] = no_hook,
    ) -> None:
        self.stream = stream
        self.to_str = to_str
        self.hook = hook
        self._iterator = (chunk for chunk in stream)

        self.head: T | None = None
        self.chunks: list[T] = []

        self.text: CustomResponseStr = CustomResponseStr()

        self.response = self.stream.response

    def __next__(self) -> T:
        chunk = next(self._iterator)

        self.head = chunk
        self.chunks += [self.head]
        head_as_str = self.to_str(self.head)
        self.text += head_as_str

        self.hook(
            CustomStreamHookArgs(self.head, self.chunks, head_as_str, self.text)
        )

        return chunk

    def close(self) -> None:
        return self.stream.close()
