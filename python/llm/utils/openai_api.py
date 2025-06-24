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
# TODO: Rewrite. Do not thread generics.
from functools import partial
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    TypeAlias,
    TypedDict,
    TypeVar,
    overload,
    override,
)

from llm.utils.context_manager import MultiroundContextManager
from llm.utils.stream import (
    CalledByFnType,
    ContentStr,
    CustomCaptureFinish,
    CustomChunkToStr,
    CustomHookArgs,
    CustomResponseStr,
    CustomStreamHandler,
    CustomStreamHook,
    CustomToolHook,
    DeltaStr,
    ReasoningContentStr,
    no_stream_hook,
    no_tool_hook,
)
from openai import NotGiven, OpenAI, Stream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params.function_definition import FunctionDefinition
from utils.logger import get_logger

logger = get_logger(__name__)


T = TypeVar('T')
U = TypeVar('U')
X = TypeVar('X')
Y = TypeVar('Y')


class ToolsHookPair(NamedTuple):
    """
    Tuple pairing tool definitions with their corresponding hook function.

    :param tools: List of function definitions for tools.
    :param hook: Hook function to process tool calls.
    :ivar toolparams: Tools in `ChatCompletionToolParam` format.
    """

    tools: list[FunctionDefinition]
    hook: CustomToolHook

    @property
    def toolparams(self) -> list[ChatCompletionToolParam]:
        return [{'type': 'function', 'function': tool} for tool in self.tools]


FinishReason: TypeAlias = Literal[  # TODO: Use.
    'stop', 'length', 'tool_calls', 'content_filter', 'function_call'
]


@CustomCaptureFinish[ChatCompletionChunk]
def capture_finish(chunk: ChatCompletionChunk) -> bool:
    """
    Determine if a chunk indicates the end of a stream.

    :param chunk: Stream chunk from OpenAI API.
    :returns: True if stream should finish, False otherwise.
    """
    # TODO: Handle refusals/connection errors.
    if finish_reason := chunk.choices[0].finish_reason:
        logger.debug(f'Finish reason: {finish_reason}')
    return bool(finish_reason)


class ExtraPayloadContents(TypedDict):
    """Additional parameters for chat completion requests."""

    frequency_penalty: float | NotGiven
    max_tokens: int | NotGiven
    presence_penalty: float | NotGiven
    response_format: ResponseFormat | NotGiven
    stop: str | NotGiven
    stream_options: ChatCompletionStreamOptionsParam | NotGiven
    temperature: float | NotGiven
    top_p: int | NotGiven
    tools: list[ChatCompletionToolParam] | NotGiven
    tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven
    logprobs: bool | NotGiven


_completion_kwargs_default: ExtraPayloadContents = {
    'frequency_penalty': NotGiven(),
    'max_tokens': NotGiven(),
    'presence_penalty': NotGiven(),
    'response_format': NotGiven(),
    'stop': NotGiven(),
    'stream_options': NotGiven(),
    'temperature': NotGiven(),
    'top_p': NotGiven(),
    'tools': NotGiven(),
    'tool_choice': NotGiven(),
    'logprobs': NotGiven(),
}


def _payload_from_partial(
    p: dict[str, Any] | ExtraPayloadContents | None = None,
    d: ExtraPayloadContents = _completion_kwargs_default,
) -> ExtraPayloadContents:
    """
    Merge partial payload with default values.

    :param part: Partial payload values.
    :param orig: Default values to fall back to.
    :returns: Merged payload dictionary.
    """
    if p is None:
        p = {}
    return {
        'frequency_penalty': p.get('frequency_penalty', d['frequency_penalty']),
        'max_tokens': p.get('max_tokens', d['max_tokens']),
        'presence_penalty': p.get('presence_penalty', d['presence_penalty']),
        'response_format': p.get('response_format', d['response_format']),
        'stop': p.get('stop', d['stop']),
        'stream_options': p.get('stream_options', d['stream_options']),
        'temperature': p.get('temperature', d['temperature']),
        'top_p': p.get('top_p', d['top_p']),
        'tools': p.get('tools', d['tools']),
        'tool_choice': p.get('tool_choice', d['tool_choice']),
        'logprobs': p.get('logprobs', d['logprobs']),
    }


class LLMProfile(Generic[T, U, X, Y]):
    """
    Profile configuration for OpenAI-style API endpoints.

    :param key: Unique identifier for the profile.
    :param api_key: API key for authentication.
    :param base_url: Base URL for API endpoint.
    :param model: Model name to use.
    :param capture_finish: Function to detect stream completion.
    :param stream_hook: Hook to process streaming chunks.
    :param tool_hook: Hook to process tool calls.
    :param completion_kwargs: Default parameters for completions.
    :param openai_kwargs: Additional parameters for OpenAI client.
    :param reasoning_field_name: Field name for reasoning content.
    """

    def __init__(
        self,
        key: str,
        api_key: str,
        base_url: str,
        model: str,
        capture_finish: CustomCaptureFinish = capture_finish,
        stream_hook: CustomStreamHook[T, U, X, Y] = no_stream_hook,
        tool_hook: CustomToolHook[T, U, X, Y] = no_tool_hook,
        completion_kwargs: dict[str, Any] | None = None,
        openai_kwargs: dict[str, Any] | None = None,
        reasoning_field_name: str = 'reasoning_content',
    ) -> None:
        self.key = key
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, **(openai_kwargs or {})
        )
        self.model = model
        self.capture_finish = capture_finish
        self.stream_hook = stream_hook
        self.tool_hook = tool_hook
        self.reasoning_field_name = reasoning_field_name

        self.completion_kwargs = _payload_from_partial(completion_kwargs)


class LLMWrapper(
    dict[
        str,
        LLMProfile[
            ChatCompletionChunk,
            T,
            ChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ],
    ]
):
    """
    Container for LLM profiles with completion methods.

    :param profiles: Initial profiles to register.
    """

    def __init__(
        self,
        *profiles: LLMProfile[
            ChatCompletionChunk,
            T,
            ChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ],
    ) -> None:
        self.profiles = {profile.key: profile for profile in profiles}

    @override
    def __setitem__(
        self,
        key: str,
        profile: LLMProfile[
            ChatCompletionChunk,
            T,
            ChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ],
        /,
    ) -> None:
        self.profiles |= {key: profile}

    @override
    def __getitem__(
        self, key: str, /
    ) -> LLMProfile[
        ChatCompletionChunk,
        T,
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    ]:
        return self.profiles[key]

    @override
    def __delitem__(self, key: str) -> None:
        del self.profiles[key]

    @overload
    def completion(
        self,
        profile_key: str,
        system_prompt: str,
        context_manager: MultiroundContextManager[T],
        /,
        stream: Literal[False],
        *,
        frequency_penalty: float | NotGiven | None = None,
        max_tokens: int | NotGiven | None = None,
        presence_penalty: float | NotGiven | None = None,
        response_format: ResponseFormat | NotGiven | None = None,
        stop: str | NotGiven | None = None,
        stream_options: ChatCompletionStreamOptionsParam
        | NotGiven
        | None = None,
        temperature: float | NotGiven | None = None,
        top_p: int | NotGiven | None = None,
        tools: list[ChatCompletionToolParam] | NotGiven | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam
        | NotGiven
        | None = None,
        logprobs: bool | NotGiven | None = None,
        specific_tools: list[ChatCompletionToolParam] | NotGiven | None = None,
        specific_tool_hook: CustomToolHook[
            ChatCompletionChunk,
            T,
            ChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ]
        | NotGiven
        | None = None,
    ) -> CustomResponseStr: ...

    @overload
    def completion(
        self,
        profile_key: str,
        system_prompt: str,
        context_manager: MultiroundContextManager[T],
        /,
        stream: Literal[True],
        *,
        frequency_penalty: float | NotGiven | None = None,
        max_tokens: int | NotGiven | None = None,
        presence_penalty: float | NotGiven | None = None,
        response_format: ResponseFormat | NotGiven | None = None,
        stop: str | NotGiven | None = None,
        stream_options: ChatCompletionStreamOptionsParam
        | NotGiven
        | None = None,
        temperature: float | NotGiven | None = None,
        top_p: int | NotGiven | None = None,
        tools: list[ChatCompletionToolParam] | NotGiven | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam
        | NotGiven
        | None = None,
        logprobs: bool | NotGiven | None = None,
        specific_tools: list[ChatCompletionToolParam] | NotGiven | None = None,
        specific_tool_hook: CustomToolHook[
            ChatCompletionChunk,
            T,
            ChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ]
        | NotGiven
        | None = None,
    ) -> CustomStreamHandler[ChatCompletionChunk]: ...

    def completion(
        self,
        profile_key: str,
        system_prompt: str,
        context_manager: MultiroundContextManager[T],
        /,
        stream: bool = False,
        *,
        frequency_penalty: float | NotGiven | None = None,
        max_tokens: int | NotGiven | None = None,
        presence_penalty: float | NotGiven | None = None,
        response_format: ResponseFormat | NotGiven | None = None,
        stop: str | NotGiven | None = None,
        stream_options: ChatCompletionStreamOptionsParam
        | NotGiven
        | None = None,
        temperature: float | NotGiven | None = None,
        top_p: int | NotGiven | None = None,
        tools: list[ChatCompletionToolParam] | NotGiven | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam
        | NotGiven
        | None = None,
        logprobs: bool | NotGiven | None = None,
        specific_tools: list[ChatCompletionToolParam] | NotGiven | None = None,
        specific_tool_hook: CustomToolHook[
            ChatCompletionChunk,
            T,
            ChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ]
        | NotGiven
        | None = None,
    ) -> str | CustomStreamHandler[ChatCompletionChunk]:
        """
        Execute chat completion with context and tool handling.

        :param profile_key: Key of LLM profile to use.
        :param system_prompt: System-level instructions.
        :param context_manager: Conversation history manager.
        :param stream: Whether to use streaming mode.
        :param kwargs: Additional completion parameters.
        :returns: Response string or stream handler.
        """
        profile = self.profiles[profile_key]
        instance_completion_kwargs: ExtraPayloadContents = {
            'frequency_penalty': frequency_penalty or NotGiven(),
            'max_tokens': max_tokens or NotGiven(),
            'presence_penalty': presence_penalty or NotGiven(),
            'response_format': response_format or NotGiven(),
            'stop': stop or NotGiven(),
            'stream_options': stream_options or NotGiven(),
            'temperature': temperature or NotGiven(),
            'top_p': top_p or NotGiven(),
            'tools': tools or NotGiven(),
            'tool_choice': tool_choice or NotGiven(),
            'logprobs': logprobs or NotGiven(),
        }

        self.context_manager = context_manager

        completion_kwargs = _payload_from_partial(
            instance_completion_kwargs, profile.completion_kwargs
        )

        # Concat `specific_tools`.
        if completion_kwargs['tools']:
            completion_kwargs['tools'] += specific_tools or []
        else:
            completion_kwargs['tools'] = specific_tools or NotGiven()

        logger.debug(
            f'Sent request to {profile.client.base_url} :: {profile.model}.'
        )
        if (
            context_manager.head is not None
            and 'content' in context_manager.head
            and context_manager.head['content'] is not None
        ):
            assert isinstance(context_manager.head['content'], str)
            logger.debug(
                f'Top of stack: {context_manager.head["content"][:200]}'
            )
        response = profile.client.chat.completions.create(
            model=profile.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                *context_manager.to_list(),
            ],
            stream=stream,
            **completion_kwargs,
        )

        # Concat `specific_tool_hook`.
        tool_hook = (
            profile.tool_hook
            if not specific_tool_hook
            else profile.tool_hook | specific_tool_hook
        )

        rfn = profile.reasoning_field_name
        if not stream:
            assert isinstance(response, ChatCompletion)
            cm = response.choices[0].message
            response_str = (
                CustomResponseStr()
                + (
                    ReasoningContentStr()
                    if not hasattr(cm, rfn)
                    else ReasoningContentStr(getattr(cm, rfn))
                )
                + ContentStr(cm.content)
            )

            found_tools = find_all_tools(response.choices[0].message)
            if found_tools:
                message = response.choices[0].message
                this_fn = self._get_completion_function(
                    profile_key=profile_key,
                    system_prompt=system_prompt,
                    stream=stream,
                    kwargs=completion_kwargs,
                )
                args = CustomHookArgs(
                    called_by=this_fn,
                    context_manager=self.context_manager,
                    head=ChatCompletionChunk(
                        id='',
                        choices=[],
                        created=0,
                        model='',
                        object='chat.completion.chunk',
                    ),
                    chunks=[],
                    head_as_str=ContentStr(),
                    text=response_str,
                    message=message,
                    tools=found_tools,
                    finished=True,
                )
                if (ret := tool_hook(args)) is not None:
                    reason, extn = ret
                    if reason is not None and extn is not None:
                        assert isinstance(extn, CustomResponseStr)
                        # TODO: Rewrite overrides.
                        response_str += reason
                        response_str += extn

            return response_str

        else:
            assert isinstance(response, Stream)
            to_str = CustomChunkToStr(
                partial(
                    self._chunk_to_str,
                    reasoning_field_name=rfn,
                )
            )
            this_fn = self._get_completion_function(
                profile_key=profile_key,
                system_prompt=system_prompt,
                stream=stream,
                kwargs=completion_kwargs,
            )
            return CustomStreamHandler(
                response,
                context_manager=self.context_manager,
                to_str=to_str,
                capture_finish=profile.capture_finish,
                stream_hook=profile.stream_hook,
                tool_hook=tool_hook,
                called_by=this_fn,
                get_tools=find_all_tools,
            )

    @staticmethod
    def _chunk_to_str(
        chunk: ChatCompletionChunk, reasoning_field_name: str
    ) -> DeltaStr:
        """
        Convert stream chunk to `DeltaStr`.

        :param chunk: Stream chunk from API.
        :param reasoning_field_name: Field name for reasoning content.
        :returns: Processed content as DeltaStr.
        """
        cd = chunk.choices[0].delta

        # For reasoning models.
        if hasattr(cd, reasoning_field_name):
            if (c := getattr(cd, reasoning_field_name)) is not None:
                assert isinstance(c, str)
                return ReasoningContentStr(c)

        if cd.content is not None:
            return ContentStr(cd.content)
        else:
            return ContentStr()

    @overload
    def _get_completion_function(
        self,
        profile_key: str,
        system_prompt: str,
        stream: Literal[False],
        kwargs: ExtraPayloadContents,
    ) -> Callable[
        [MultiroundContextManager, dict[str, Any] | None], CustomResponseStr
    ]: ...

    @overload
    def _get_completion_function(
        self,
        profile_key: str,
        system_prompt: str,
        stream: Literal[True],
        kwargs: ExtraPayloadContents,
    ) -> Callable[
        [MultiroundContextManager, dict[str, Any] | None],
        Iterator[ChatCompletionChunk],
    ]: ...

    def _get_completion_function(
        self,
        profile_key: str,
        system_prompt: str,
        stream: bool,
        kwargs: ExtraPayloadContents,
    ) -> CalledByFnType[ChatCompletionChunk]:
        """
        Create a reusable completion function closure.

        :param profile_key: Key of LLM profile to use.
        :param system_prompt: System-level instructions.
        :param stream: Whether to use streaming mode.
        :param kwargs: Base completion parameters.
        :returns: Configured completion function.
        """

        def _completion_function(
            context_manager: MultiroundContextManager,
            _mask_kwargs: dict[str, Any] | None = None,
        ) -> CustomResponseStr | Iterator[ChatCompletionChunk]:
            if 'stream' in (mask_kwargs := _mask_kwargs or {}):
                raise RuntimeError('Do not pass `stream`.')

            # Allow temporary settings override.
            completion = self.completion(
                profile_key,
                system_prompt,
                context_manager,
                stream=stream,
                **_payload_from_partial(mask_kwargs, kwargs),
            )

            if isinstance(completion, CustomResponseStr):
                return completion
            else:
                return completion._iterator

        return _completion_function


@overload
def find_last_tool(
    chunks: Iterable[ChatCompletionChunk],
    nonu: Literal[False],
) -> tuple[ChatCompletionMessageToolCall, int] | None: ...


@overload
def find_last_tool(
    chunks: Iterable[ChatCompletionChunk],
    nonu: Literal[True],
) -> ChatCompletionMessageToolCall | None: ...


def find_last_tool(
    chunks: Iterable[ChatCompletionChunk],
    nonu: bool = True,
) -> (
    tuple[ChatCompletionMessageToolCall, int]
    | ChatCompletionMessageToolCall
    | None
):
    """
    Locate the last tool call in a stream of chunks.

    :param chunks: Stream chunks to analyze.
    :param nonu: Whether to return position index.
    :returns: Tool call and optional position, or None.
    """
    deltas = [c.choices[0].delta for c in chunks]

    last_tool_start_index = len(deltas) - 1
    for delta in reversed(deltas):
        if (
            (tc := delta.tool_calls) is not None
            and len(tc) > 0
            and (tool_id := tc[0].id) is not None
            and ((partial_fn := tc[0].function) is not None)
            and ((tool_function_name := partial_fn.name) is not None)
            and (tool_type := tc[0].type) is not None
        ):
            the_tool_id = tool_id
            the_tool_function_name = tool_function_name
            the_tool_type = tool_type
            break
        last_tool_start_index -= 1
    else:
        return None

    concat_function_arguments = ''
    for delta in deltas[last_tool_start_index:]:
        if (
            (tc := delta.tool_calls) is not None
            and len(tc) > 0
            and (delta_fn := tc[0].function) is not None
        ):
            concat_function_arguments += delta_fn.arguments or ''

    the_function = Function(
        arguments=concat_function_arguments,
        name=the_tool_function_name,
    )

    ccmtc = ChatCompletionMessageToolCall(
        id=the_tool_id,
        function=the_function,
        type=the_tool_type,
    )

    return ccmtc if nonu else ccmtc, last_tool_start_index


# find_all_tools: GetToolsFnType
def find_all_tools(
    chunks: list[ChatCompletionChunk] | ChatCompletionMessage,
) -> list[ChatCompletionMessageToolCall]:
    """
    Extract all tool calls from message or chunks.

    :param chunks: Message or chunk list to analyze.
    :returns: List of extracted tool calls.
    """
    if isinstance(chunks, list):
        tools, n = [], len(chunks)
        while (lc := find_last_tool(chunks[:n], False)) is not None:
            tool, n = lc
            tools += [tool]
    else:
        tools = chunks.tool_calls or []

    logger.debug(
        'Found tools:\n' + '\n'.join(str(tool.function) for tool in tools)
    )

    return tools
