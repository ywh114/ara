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
X = TypeVar('X')
Y = TypeVar('Y')


class ReasoningContentStr(str):
    """A string subclass that represents reasoning content."""


class ContentStr(str):
    """A string subclass that represents regular content."""


class ToolBlurbStr(str):
    """A string subclass that represents a tool call."""


DeltaStr: TypeAlias = ReasoningContentStr | ContentStr | ToolBlurbStr
"""
Type alias for string deltas in streamed responses.

`ReasoningContentStr` | `ContentStr` | `ToolBlurbStr`
"""


# FIXME: Remove redundancy.
class CustomResponseStr(str):
    """
    Enhanced string class with specialized content tracking.

    Tracks different content types separately while maintaining full response
    text.

    :ivar reasoning_content: Accumulated reasoning thought content.
    :ivar tool_blurbs: Accumulated tool call descriptions.
    :ivar content: Regular response content.
    :ivar content_with_reasoning: Combined reasoning + regular content.
    :ivar content_with_tool_blurbs: Combined tool blurbs + regular content.
    """

    reasoning_content: ReasoningContentStr
    content: ContentStr
    tool_blurbs: ToolBlurbStr
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
        """
        Create new CustomResponseStr instance.

        :param value: Base string value.
        :param reasoning_content: Initial reasoning content.
        :param content: Initial regular content.
        :param tool_blurbs: Initial tool call descriptions.
        :param content_with_reasoning: Initial combined reasoning+content.
        :param content_with_tool_blurbs: Initial combined tool blurbs+content.
        :return: New instance with initialized attributes.
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
        Concatenate strings while maintaining content type tracking.

        :param other: String to add (regular str or specialized type).
        :return: New CustomResponseStr with updated attributes.
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
        In-place string concatenation with content type tracking.

        :param other: String to add (regular str or specialized type).
        :return: Updated self with new content.
        """
        return self.__add__(other)


CalledByFnType: TypeAlias = Callable[
    [MultiroundContextManager, dict[str, Any] | None],
    CustomResponseStr | Iterator[T],
]
"""
Type alias for originator functions.

Signature:
    `(context_manager: MultiroundContextManager, 
     kwargs: dict[str, Any] | None) -> CustomResponseStr | Iterator[T]`
"""

GetToolsFnType: TypeAlias = Callable[[list[T]], list[V]]
"""
Type alias for tool extraction functions.

Signature: 
    `(chunks: list[T]) -> list[V]`
"""


@dataclass
class CustomHookArgs(Generic[T, U, X, Y]):
    """
    Container for stream processing state.

    :ivar called_by: Originator function that started the stream.
    :ivar context_manager: Conversation context manager.
    :ivar head: Most recent chunk from stream.
    :ivar chunks: All accumulated chunks.
    :ivar head_as_str: String representation of current chunk.
    :ivar text: Aggregated response content.
    :ivar message: Whole message.
    :ivar finished: Stream completion flag.
    """

    called_by: CalledByFnType[T]
    context_manager: MultiroundContextManager[U]
    head: T
    chunks: list[T]
    head_as_str: DeltaStr
    text: CustomResponseStr
    message: X | None
    tools: list[Y] | None
    finished: bool


CustomChunkToStrFnType: TypeAlias = Callable[[T], DeltaStr]
"""
Type alias for chunk conversion functions.

Signature:
    `(chunk: T) -> DeltaStr`
"""

CustomStreamHookFnType: TypeAlias = Callable[[CustomHookArgs[T, U, X, Y]], None]
"""
Type alias for stream hook functions.

Signature:
    `(args: CustomHookArgs[T]) -> None`
"""


ToolCallChainExtension: TypeAlias = tuple[ToolBlurbStr, Iterator[T]]
ToolCallExtension: TypeAlias = tuple[ToolBlurbStr, CustomResponseStr]

CustomToolHookFnType: TypeAlias = Callable[
    [CustomHookArgs[T, U, X, Y]],
    ToolCallExtension | ToolCallChainExtension[T] | None,
]
"""
Type alias for tool hook functions.

Signature: 
    `(args: CustomHookArgs[T, U], 
     tools: list[V]) -> ToolCallExtension | ToolCallChainExtension[T] | None`
"""

CustomCaptureFinishFnType: TypeAlias = Callable[[T], bool]
"""
Type alias for stream completion detection.

Signature: 
    `(chunk: T) -> bool`
"""


class CustomChunkToStr(Generic[T]):
    """
    Decorator for chunk-to-string conversion functions.

    :param fn: Conversion function to wrap.
    """

    def __init__(self, fn: CustomChunkToStrFnType[T]) -> None:
        self.to_str = fn
        update_wrapper(self, fn)

    def __call__(self, chunk: T) -> DeltaStr:
        return self.to_str(chunk)


# XXX: Wrappers are currently useless. Probably could be used to avoid redundant
# operations.


class CustomStreamHook(Generic[T, U, X, Y]):
    """Container for sequence of stream hook functions."""

    def __init__(self, *fns: CustomStreamHookFnType[T, U, X, Y]) -> None:
        """
        Initialize with hook functions.

        :param fns: Stream hook functions to register.
        """
        self.hooks = fns

    def __call__(self, args: CustomHookArgs[T, U, X, Y]) -> None:
        """
        Execute all registered hooks in sequence.

        :param args: Current stream processing state.
        """
        for hook in self.hooks:
            hook(args)

    def __or__(
        self, other: 'CustomStreamHook[T, U, X, Y]'
    ) -> 'CustomStreamHook[T, U, X, Y]':
        """
        Combine hook sequences.

        :param other: Additional hooks to merge
        :return: New `CustomStreamHook` with combined hooks
        """
        return CustomStreamHook(*self.hooks, *other.hooks)


class CustomToolHook(Generic[T, U, X, Y]):
    """Container for sequence of tool hook functions."""

    def __init__(self, *fns: CustomToolHookFnType[T, U, X, Y]) -> None:
        """
        Initialize with tool hook functions.

        :param fns: Tool hook functions to register.
        """
        self.hooks = fns

    # FIXME: Unify this with top-level `create_tool_hook`.
    def __call__(
        self, args: CustomHookArgs[T, U, X, Y]
    ) -> ToolCallExtension | ToolCallChainExtension[T] | None:
        """
        Execute registered hooks until one returns non-None.

        :param args: Current stream processing state.
        :param tools: Extracted tools from stream.
        :return: First non-None result from hooks or None.
        """
        for ret in (hook(args) for hook in self.hooks):
            if ret is not None:
                return ret
        else:
            return None

    def __or__(
        self, other: 'CustomToolHook[T, U, X, Y]'
    ) -> 'CustomToolHook[T, U, X, Y]':
        """
        Combine tool hook sequences.

        :param other: Additional tool hooks to merge.
        :return: New `CustomToolHook` with combined hooks.
        """
        return CustomToolHook(*self.hooks, *other.hooks)


# TODO: Add multiple end states for refusal etc.
class CustomCaptureFinish(Generic[T]):
    """Decorator for stream completion detection."""

    def __init__(self, fn: CustomCaptureFinishFnType) -> None:
        """
        Initialize with detection function.

        :param fn: Stream completion detector.
        """
        self.capture_finish = fn
        update_wrapper(self, fn)

    def __call__(self, chunk: T) -> bool:
        """
        Determine if chunk indicates stream completion.

        :param chunk: Input data chunk.
        :return: True if stream should terminate.
        """
        return self.capture_finish(chunk)


@CustomChunkToStr
def to_str_not_implemented(_) -> DeltaStr:
    """Default chunk converter (returns empty `ContentStr`)."""
    return ContentStr()


@CustomStreamHook
def no_stream_hook(_: CustomHookArgs[T, U, X, Y]) -> None:
    """Default no-op stream hook."""


@CustomToolHook
def no_tool_hook(*_) -> None:
    """Default no-op stream hook."""


@CustomCaptureFinish
def no_capture(_) -> Literal[False]:
    """Default ignore stream-finish capture."""
    return False


# TODO: Stop threading `context_manager`; get it into `CustomHookArgs` another
# way. Feasability?
class CustomStreamHandler(Iterator[T]):
    """
    Handles streaming responses with content tracking and hook execution.

    :ivar stream: Underlying response stream.
    :ivar context_manager: Conversation context.
    :ivar to_str: Chunk-to-string converter.
    :ivar capture_finish: Stream completion detector.
    :ivar stream_hook: Stream processing hooks.
    :ivar tool_hook: Tool processing hooks.
    :ivar called_by: Originator function.
    :ivar get_tools: Tool extraction function.
    :ivar head: Current chunk being processed.
    :ivar chunks: Accumulated chunks.
    :ivar head_as_str: Stringified current chunk.
    :ivar chunks_as_str: Stringified accumulated chunks.
    :ivar text: Aggregated response content.
    :ivar stream_finished: Stream completion flag.
    :ivar http_response: Original HTTP response.
    """

    def __init__(
        self,
        stream: Stream[T],
        context_manager: MultiroundContextManager[U],
        to_str: CustomChunkToStr[T] = to_str_not_implemented,
        capture_finish: CustomCaptureFinish[T] = no_capture,
        stream_hook: CustomStreamHook[T, U, X, Y] = no_stream_hook,
        tool_hook: CustomToolHook[T, U, X, Y] = no_tool_hook,
        called_by: CalledByFnType[T] = lambda *_: iter(()),
        get_tools: GetToolsFnType[T, V] = lambda *_: [],
    ) -> None:
        """
        Initialize stream handler.

        :param stream: Response stream to handle.
        :param context_manager: Conversation context.
        :param to_str: Chunk converter (default: empty content).
        :param capture_finish: Completion detector (default: no detection).
        :param stream_hook: Processing hooks (default: no-op).
        :param tool_hook: Tool hooks (default: no-op).
        :param called_by: Originator function (default: empty iterator).
        :param get_tools: Tool extractor (default: empty list).
        """
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

    @property
    def chunks_as_str(self) -> CustomResponseStr:
        """Points to `self.text`."""
        return self.text

    def _clear(self) -> None:
        """Reset accumulated state after tool chain extension."""
        self.chunks = []
        self.stream_finished = False

    def __next__(self) -> T:
        """
        Get next chunk from stream.

        :return: Next data chunk.
        :raises StopIteration: When stream exhausted.
        """
        chunk = next(self._iterator)

        self.stream_finished = self.capture_finish(chunk)

        self.head = chunk
        self.chunks += [self.head]
        self.head_as_str = self.to_str(self.head)
        self.text += self.head_as_str

        args = CustomHookArgs(
            called_by=self.called_by,
            context_manager=self.context_manager,
            head=self.head,
            chunks=self.chunks,
            head_as_str=self.head_as_str,
            text=self.text,
            message=None,
            tools=None,
            finished=self.stream_finished,
        )

        # Run stream hooks.
        self.stream_hook(args)

        if self.stream_finished:
            tools = self.get_tools(self.chunks)
            args.tools = tools
            # Run finish hooks.
            if (ret := self.tool_hook(args)) is not None:
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
        """
        Consume entire stream and return aggregated content.

        :return: Final aggregated response content.
        """
        _ = tuple(_ for _ in self)
        return self.text
