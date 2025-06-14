#!/usr/bin/env python3
from dataclasses import dataclass
from functools import update_wrapper
from itertools import chain
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
)

from httpx import Response
from openai import Stream

T = TypeVar('T')


class ReasoningContentStr(str):
    """A string subclass that represents reasoning content."""


class ContentStr(str):
    """A string subclass that represents regular content."""


class ToolBlurbStr(str):
    """A string subclass that represents a tool call."""


DeltaStr: TypeAlias = ReasoningContentStr | ContentStr | ToolBlurbStr
"""
Type alias for string deltas in streamed responses.
"""


class CustomResponseStr(str):
    """
    A string subclass with additional attributes for reasoning and content.

    :ivar reasoning_content: A string representing reasoning content.
    :ivar content: A string representing regular content.
    """

    reasoning_content: ReasoningContentStr
    tool_blurbs: ToolBlurbStr
    content: ContentStr
    content_with_tool_blurbs: str

    def __new__(
        cls,
        value: str = '',
        reasoning_content: str = '',
        tool_blurb_content: str = '',
        content: str = '',
        content_with_tool_blurbs: str = '',
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
        instance.tool_blurbs = ToolBlurbStr(tool_blurb_content)
        instance.content_with_tool_blurbs = content_with_tool_blurbs
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
            updated.tool_blurbs = ToolBlurbStr(
                updated.tool_blurbs + other.tool_blurbs
            )
            updated.content_with_tool_blurbs = (
                updated.content_with_tool_blurbs
                + other.content_with_tool_blurbs
            )
        elif isinstance(other, ReasoningContentStr):
            updated.reasoning_content = ReasoningContentStr(
                updated.reasoning_content + other
            )
        elif isinstance(other, ToolBlurbStr):
            updated.tool_blurbs = ToolBlurbStr(updated.tool_blurbs + other)
            updated.content_with_tool_blurbs += other
        elif isinstance(other, (ContentStr, str)):  # Catch other `str`.
            updated.content = ContentStr(updated.content + other)
            updated.content_with_tool_blurbs += other

        return updated

    def __iadd__(self, other: str | DeltaStr | Self) -> Self:
        """
        In-place addition for `ResponseStr`.
        Treat other all other subclasses of `str` as `ContentStr`.

        :param other: The value to add.
        :return: Updated instance with concatenated values.
        """
        return self.__add__(other)


CalledByFnType: TypeAlias = Callable[
    [str, dict[str, Any] | None], CustomResponseStr | Iterator[T]
]


@dataclass
class CustomHookArgs(Generic[T]):
    """
    Container for stream processing arguments.

    Attributes:
        called_by: The function this was called by.
        head: Most recent chunk from stream.
        chunks: All accumulated chunks.
        head_as_str: String representation of head chunk.
        text: Aggregated response content.
    """

    called_by: CalledByFnType[T]
    head: T
    chunks: list[T]
    head_as_str: DeltaStr
    text: CustomResponseStr
    finished: bool


CustomChunkToStrFnType: TypeAlias = Callable[[T], DeltaStr]
"""
Type alias for chunk conversion functions.

Signature:
    `(chunk: T) -> DeltaStr`
"""

CustomStreamHookFnType: TypeAlias = Callable[[CustomHookArgs[T]], None]
"""
Type alias for stream hook functions.

Signature:
    `(cha: CustomHookArgs[T]) -> None`
"""

ToolCallChainer: TypeAlias = tuple[ToolBlurbStr, Iterator[T]]

CustomFinishHookFnType: TypeAlias = Callable[
    [CustomHookArgs[T]], ToolCallChainer[T] | None
]
"""
Type alias for finish hook functions. Can extend via tool calls.

Signature:
    `(cha: CustomHookArgs[T]) -> ToolCallChainer[T] | None`
"""

CustomCaptureFinishFnType: TypeAlias = Callable[[T], bool]
"""
Type alias for determining stream finish before `StopIteration` is raised.

Signature:
    `(chunk: T) -> bool`
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
    def __init__(self, *fns: CustomStreamHookFnType[T]) -> None:
        self.hooks = fns

    def __call__(self, args: CustomHookArgs[T]) -> None:
        for hook in self.hooks:
            hook(args)

    def __or__(self, other: 'CustomStreamHook[T]') -> 'CustomStreamHook[T]':
        return CustomStreamHook(*self.hooks, *other.hooks)


class CustomFinishHook(Generic[T]):
    def __init__(self, *fns: CustomFinishHookFnType[T]) -> None:
        self.hooks = fns

    def __call__(
        self, args: CustomHookArgs
    ) -> Iterable[ToolCallChainer[T] | None]:
        return (hook(args) for hook in self.hooks)


class CustomCaptureFinish(Generic[T]):
    def __init__(self, fn: CustomCaptureFinishFnType) -> None:
        self.capture_finish = fn
        update_wrapper(self, fn)

    def __call__(self, chunk: T) -> bool:
        return self.capture_finish(chunk)


@CustomChunkToStr
def to_str_not_implemented(_) -> DeltaStr:
    """Default chunk converter (returns empty `ContentStr`)."""
    return ContentStr()


@CustomStreamHook
def no_stream_hook(_: CustomHookArgs[T]) -> None:
    """Default no-op stream hook."""


@CustomFinishHook
def no_finish_hook(_: CustomHookArgs[T]) -> None:
    """Default no-op stream hook."""


@CustomCaptureFinish
def no_capture(_) -> Literal[False]:
    """Default ignore stream-finish capture."""
    return False


class CustomStreamHandler(Iterator[T]):
    def __init__(
        self,
        stream: Stream[T],
        to_str: CustomChunkToStr[T] = to_str_not_implemented,
        capture_finish: CustomCaptureFinish[T] = no_capture,
        stream_hook: CustomStreamHook[T] = no_stream_hook,
        finish_hook: CustomFinishHook[T] = no_finish_hook,
        called_by: CalledByFnType[T] = lambda *_: iter(()),
    ) -> None:
        # Not cleared.
        self.stream = stream
        self.to_str = to_str
        self.capture_finish = capture_finish
        self.stream_hook = stream_hook
        self.finish_hook = finish_hook
        self.called_by = called_by
        self._iterator = (chunk for chunk in stream)

        # Destroyed on clear.
        self.head: T | None = None
        self.chunks: list[T] = []
        self.head_as_str: DeltaStr = ContentStr()
        self.text: CustomResponseStr = CustomResponseStr()
        self.stream_finished: bool = False
        self.http_response: Response = self.stream.response

    def _clear(self) -> None:
        self.head: T | None = None
        self.chunks: list[T] = []
        self.head_as_str: DeltaStr = ContentStr()
        self.text: CustomResponseStr = CustomResponseStr()
        self.stream_finished: bool = False
        self.http_response: Response = self.stream.response

    def __next__(self) -> T:
        chunk = next(self._iterator)

        self.stream_finished = self.capture_finish(chunk)

        self.head = chunk
        self.chunks += [self.head]
        self.head_as_str = self.to_str(self.head)
        self.text += self.head_as_str

        ha = CustomHookArgs(
            called_by=self.called_by,
            head=self.head,
            chunks=self.chunks,
            head_as_str=self.head_as_str,
            text=self.text,
            finished=self.stream_finished,
        )

        # Run stream hooks.
        self.stream_hook(ha)

        if self.stream_finished:
            # Run finish hooks.
            for reason, extn in (
                a2 for a2 in self.finish_hook(ha) if a2 is not None
            ):
                if reason is not None and extn is not None:
                    self.text += reason
                    self._iterator = chain(self._iterator, extn)

        return chunk

    def close(self) -> None:
        """Closes the underlying stream."""
        return self.stream.close()

    def exhaust(self) -> CustomResponseStr:
        """Exhaust the iterator."""
        _ = tuple(_ for _ in self)
        return self.text
