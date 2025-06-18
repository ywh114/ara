#!/usr/bin/env python3
# TODO: Write ConversationManager for single multiround.
# TODO: Add conversations to db.
from functools import partial
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    TypeAlias,
    TypedDict,
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
    CustomToolHook,
    CustomResponseStr,
    CustomStreamHandler,
    CustomStreamHook,
    DeltaStr,
    ReasoningContentStr,
    no_tool_hook,
    no_stream_hook,
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
from utils.logger import get_logger

logger = get_logger(__name__)

FinishReason: TypeAlias = Literal[
    'stop', 'length', 'tool_calls', 'content_filter', 'function_call'
]

ChatCompletionToolParams: TypeAlias = Iterable[ChatCompletionToolParam]


@CustomCaptureFinish
def capture_finish(chunk: ChatCompletionChunk) -> bool:
    # TODO: Handle refusals/connection errors.
    return bool(chunk.choices[0].finish_reason)


class ExtraPayloadContents(TypedDict):
    frequency_penalty: float | NotGiven
    max_tokens: int | NotGiven
    presence_penalty: float | NotGiven
    response_format: ResponseFormat | NotGiven
    stop: str | NotGiven
    stream_options: ChatCompletionStreamOptionsParam | NotGiven
    temperature: float | NotGiven
    top_p: int | NotGiven
    tools: ChatCompletionToolParams | NotGiven
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
    part: dict[str, Any] | ExtraPayloadContents | None = None,
    orig: ExtraPayloadContents = _completion_kwargs_default,
) -> ExtraPayloadContents:
    if part is None:
        part = {}
    return {
        'frequency_penalty': part.setdefault('frequency_penalty', NotGiven())
        or orig['frequency_penalty'],
        'max_tokens': part.setdefault('max_tokens', NotGiven())
        or orig['max_tokens'],
        'presence_penalty': part.setdefault('presence_penalty', NotGiven())
        or orig['presence_penalty'],
        'response_format': part.setdefault('response_format', NotGiven())
        or orig['response_format'],
        'stop': part.setdefault('stop', NotGiven()) or orig['stop'],
        'stream_options': part.setdefault('stream_options', NotGiven())
        or orig['stream_options'],
        'temperature': part.setdefault('temperature', NotGiven())
        or orig['temperature'],
        'top_p': part.setdefault('top_p', NotGiven()) or orig['top_p'],
        'tools': part.setdefault('tools', NotGiven()) or orig['tools'],
        'tool_choice': part.setdefault('tool_choice', NotGiven())
        or orig['tool_choice'],
        'logprobs': part.setdefault('logprobs', NotGiven()) or orig['logprobs'],
    }


class LLMProfile:
    """Profile for OpenAI-style API."""

    def __init__(
        self,
        key: str,
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str,
        capture_finish: CustomCaptureFinish = capture_finish,
        stream_hook: CustomStreamHook = no_stream_hook,
        tool_hook: CustomToolHook = no_tool_hook,
        completion_kwargs: dict[str, Any] | None = None,
        openai_kwargs: dict[str, Any] | None = None,
        reasoning_field_name: str = 'reasoning_content',
    ) -> None:
        self.key = key
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, **(openai_kwargs or {})
        )
        self.model = model
        self.system_prompt = system_prompt
        self.capture_finish = capture_finish
        self.stream_hook = stream_hook
        self.tool_hook = tool_hook
        self.reasoning_field_name = reasoning_field_name

        self.completion_kwargs = _payload_from_partial(completion_kwargs)


class LLMWrapper(dict[str, LLMProfile]):
    def __init__(self, *profiles: LLMProfile) -> None:
        self.profiles = {profile.key: profile for profile in profiles}

    @override
    def __setitem__(self, key: str, profile: LLMProfile, /) -> None:
        self.profiles |= {key: profile}

    @override
    def __getitem__(self, key: str, /) -> LLMProfile:
        return self.profiles[key]

    @override
    def __delitem__(self, key: str) -> None:
        del self.profiles[key]

    @overload
    def completion(
        self,
        profile_key: str,
        context_manager: MultiroundContextManager,
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
        tools: ChatCompletionToolParams | NotGiven | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam
        | NotGiven
        | None = None,
        logprobs: bool | NotGiven | None = None,
        specific_tools: ChatCompletionToolParams | NotGiven | None = None,
        specific_tool_hook: CustomToolHook | NotGiven | None = None,
    ) -> CustomResponseStr: ...

    @overload
    def completion(
        self,
        profile_key: str,
        context_manager: MultiroundContextManager,
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
        tools: ChatCompletionToolParams | NotGiven | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam
        | NotGiven
        | None = None,
        logprobs: bool | NotGiven | None = None,
        specific_tools: ChatCompletionToolParams | NotGiven | None = None,
        specific_tool_hook: CustomToolHook  # TODO: Generics
        | NotGiven
        | None = None,
    ) -> CustomStreamHandler[ChatCompletionChunk]: ...

    def completion(
        self,
        profile_key: str,
        context_manager: MultiroundContextManager,
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
        tools: ChatCompletionToolParams | NotGiven | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam
        | NotGiven
        | None = None,
        logprobs: bool | NotGiven | None = None,
        specific_tools: ChatCompletionToolParams | NotGiven | None = None,
        specific_tool_hook: CustomToolHook | NotGiven | None = None,
    ) -> str | CustomStreamHandler[ChatCompletionChunk]:
        _tools = list(tools or []) + list(specific_tools or [])

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
            'tools': _tools or NotGiven(),
            'tool_choice': tool_choice or NotGiven(),
            'logprobs': logprobs or NotGiven(),
        }

        self.context_manager = context_manager

        completion_kwargs = _payload_from_partial(
            instance_completion_kwargs, profile.completion_kwargs
        )

        logger.debug(
            f'Sent request to {profile.client.base_url} {profile.model}.'
        )
        response = profile.client.chat.completions.create(
            model=profile.model,
            messages=[
                {'role': 'system', 'content': profile.system_prompt},
                *context_manager.tolist(),
            ],
            stream=stream,
            **completion_kwargs,
        )

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
                    stream=stream,
                    kwargs=completion_kwargs,
                )
                args = CustomHookArgs(
                    called_by=this_fn,
                    context_manager=self.context_manager,
                    head=None,
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
        stream: Literal[False],
        kwargs: ExtraPayloadContents,
    ) -> Callable[
        [MultiroundContextManager, dict[str, Any] | None], CustomResponseStr
    ]: ...

    @overload
    def _get_completion_function(
        self,
        profile_key: str,
        stream: Literal[True],
        kwargs: ExtraPayloadContents,
    ) -> Callable[
        [MultiroundContextManager, dict[str, Any] | None],
        Iterator[ChatCompletionChunk],
    ]: ...

    def _get_completion_function(
        self, profile_key: str, stream: bool, kwargs: ExtraPayloadContents
    ) -> CalledByFnType[ChatCompletionChunk]:
        def _completion_function(
            context_manager: MultiroundContextManager,
            _mask_kwargs: dict[str, Any] | None = None,
        ) -> CustomResponseStr | Iterator[ChatCompletionChunk]:
            if 'stream' in (mask_kwargs := _mask_kwargs or {}):
                raise RuntimeError('Do not pass `stream`.')

            # Allow temporary settings override.
            completion = self.completion(
                profile_key,
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
    if isinstance(chunks, list):
        tools, n = [], len(chunks)
        while (lc := find_last_tool(chunks[:n], False)) is not None:
            tool, n = lc
            tools += [tool]
    else:
        tools = chunks.tool_calls or []

    return tools
