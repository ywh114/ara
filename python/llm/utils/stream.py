#!/usr/bin/env python3
from dataclasses import dataclass
from functools import update_wrapper
from itertools import chain
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
)

from httpx import Response
from llm.utils.context_manager import MultiroundContextManager
from openai import Stream

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


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


# TODO: Improve. Remove redundancy.
class CustomResponseStr(str):
    """
    A string subclass with additional attributes for reasoning and content.

    :ivar reasoning_content: A string representing reasoning content.
    :ivar content: A string representing regular content.
    """

    reasoning_content: ReasoningContentStr
    tool_blurbs: ToolBlurbStr
    content: ContentStr
    content_with_reasoning: str
    content_with_tool_blurbs: str

    def __new__(
        cls,
        value: str = '',
        reasoning_content: str = '',
        content: str = '',
        tool_blurbs: str = '',
        content_with_reasoning: str = '',
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
        instance.tool_blurbs = ToolBlurbStr(tool_blurbs)
        instance.content_with_reasoning = content_with_reasoning
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
        updated = self.__class__(
            new_str,
            self.reasoning_content,
            self.content,
            self.tool_blurbs,
            self.content_with_reasoning,
            self.content_with_tool_blurbs,
        )

        if isinstance(other, self.__class__):
            updated.reasoning_content = ReasoningContentStr(
                updated.reasoning_content + other.reasoning_content
            )
            updated.content = ContentStr(updated.content + other.content)
            updated.tool_blurbs = ToolBlurbStr(
                updated.tool_blurbs + other.tool_blurbs
            )
            updated.content_with_reasoning = (
                updated.reasoning_content + updated.content
            )
            updated.content_with_tool_blurbs = (
                updated.content_with_tool_blurbs
                + other.content_with_tool_blurbs
            )
        elif isinstance(other, ReasoningContentStr):
            updated.reasoning_content = ReasoningContentStr(
                updated.reasoning_content + other
            )
            updated.content_with_reasoning += other
        elif isinstance(other, ToolBlurbStr):
            updated.tool_blurbs = ToolBlurbStr(updated.tool_blurbs + other)
            updated.content_with_tool_blurbs += other
        elif isinstance(other, (ContentStr, str)):  # Catch other `str`.
            updated.content = ContentStr(updated.content + other)
            updated.content_with_reasoning += other
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
    [MultiroundContextManager, dict[str, Any] | None],
    CustomResponseStr | Iterator[T],
]

GetToolsFnType: TypeAlias = Callable[[list[T]], list[V]]


@dataclass
class CustomHookArgs(Generic[T, U]):
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
    context_manager: MultiroundContextManager[U]
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

CustomStreamHookFnType: TypeAlias = Callable[[CustomHookArgs[T, U]], None]
"""
Type alias for stream hook functions.

Signature:
    `(cha: CustomHookArgs[T]) -> None`
"""


ToolCallChainExtension: TypeAlias = tuple[ToolBlurbStr, Iterator[T]]
ToolCallExtension: TypeAlias = tuple[ToolBlurbStr, CustomResponseStr]

CustomToolHookFnType: TypeAlias = Callable[
    [CustomHookArgs[T, U], list[V]],
    ToolCallExtension | ToolCallChainExtension[T] | None,
]
"""
Type alias for tool hook functions. Can extend via tool calls.

Signature:
    `(cha: CustomHookArgs[T]) -> ToolCallChainExtension[T] | None`
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


# XXX: Wrappers are currently useless. Probably could be used to avoid redundant
# operations.


class CustomStreamHook(Generic[T, U]):
    def __init__(self, *fns: CustomStreamHookFnType[T, U]) -> None:
        self.hooks = fns

    def __call__(self, args: CustomHookArgs[T, U]) -> None:
        for hook in self.hooks:
            hook(args)

    def __or__(
        self, other: 'CustomStreamHook[T, U]'
    ) -> 'CustomStreamHook[T, U]':
        return CustomStreamHook(*self.hooks, *other.hooks)


class CustomToolHook(Generic[T, U, V]):
    def __init__(self, *fns: CustomToolHookFnType[T, U, V]) -> None:
        self.hooks = fns

    def __call__(
        self, ha: CustomHookArgs[T, U], tools: list[V]
    ) -> ToolCallExtension | ToolCallChainExtension[T] | None:
        for ret in (hook(ha, tools) for hook in self.hooks):
            if ret is not None:
                return ret
        else:
            return None

    def __or__(
        self, other: 'CustomToolHook[T, U, V]'
    ) -> 'CustomToolHook[T, U, V]':
        return CustomToolHook(*self.hooks, *other.hooks)


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
def no_stream_hook(_: CustomHookArgs[T, U]) -> None:
    """Default no-op stream hook."""


@CustomToolHook
def no_tool_hook(*_) -> None:
    """Default no-op stream hook."""


@CustomCaptureFinish
def no_capture(_) -> Literal[False]:
    """Default ignore stream-finish capture."""
    return False


class CustomStreamHandler(Iterator[T]):
    def __init__(
        self,
        stream: Stream[T],
        context_manager: MultiroundContextManager[U],
        to_str: CustomChunkToStr[T] = to_str_not_implemented,
        capture_finish: CustomCaptureFinish[T] = no_capture,
        stream_hook: CustomStreamHook[T, U] = no_stream_hook,
        tool_hook: CustomToolHook[T, U, V] = no_tool_hook,
        called_by: CalledByFnType[T] = lambda *_: iter(()),
        get_tools: GetToolsFnType[T, V] = lambda *_: [],
    ) -> None:
        self.stream = stream
        self.context_manager = context_manager
        self.to_str = to_str
        self.capture_finish = capture_finish
        self.stream_hook = stream_hook
        self.tool_hook = tool_hook
        self.called_by = called_by
        self.get_tools = get_tools
        self._iterator = (chunk for chunk in stream)

        self.head: T | None = None
        self.chunks: list[T] = []  # Destroyed on clear.

        self.head_as_str: DeltaStr = ContentStr()
        self.text: CustomResponseStr = CustomResponseStr()

        self.stream_finished: bool = False  # Reset on clear.
        self.http_response: Response = self.stream.response

    def _clear(self) -> None:
        self.chunks = []
        self.stream_finished = False

    def __next__(self) -> T:
        chunk = next(self._iterator)

        self.stream_finished = self.capture_finish(chunk)

        self.head = chunk
        self.chunks += [self.head]
        self.head_as_str = self.to_str(self.head)
        self.text += self.head_as_str

        ha = CustomHookArgs(
            called_by=self.called_by,
            context_manager=self.context_manager,
            head=self.head,
            chunks=self.chunks,
            head_as_str=self.head_as_str,
            text=self.text,
            finished=self.stream_finished,
        )

        # Run stream hooks.
        self.stream_hook(ha)

        if self.stream_finished:
            tools = self.get_tools(self.chunks)
            # Run finish hooks.
            if (ret := self.tool_hook(ha, tools)) is not None:
                reason, extn = ret
                if reason is not None and extn is not None:
                    assert isinstance(extn, Iterator)  # Always true.
                    self.text += reason
                    self._iterator = chain(self._iterator, extn)
                    self._clear()

        return chunk

    def close(self) -> None:
        """Closes the underlying stream."""
        return self.stream.close()

    def exhaust(self) -> CustomResponseStr:
        """Exhaust the iterator."""
        _ = tuple(_ for _ in self)
        return self.text
