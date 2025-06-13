#!/usr/bin/env python3
# TODO: Write ConversationManager for multiround.
# TODO: Add conversations to db.
from functools import partial
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    TypeAlias,
    TypedDict,
    overload,
    override,
)

from llm.utils.stream import (
    CompletionFn,
    ContentStr,
    CustomChunkToStr,
    CustomResponseStr,
    CustomStreamHandler,
    CustomStreamHook,
    DeltaStr,
    ReasoningContentStr,
    no_hook,
)
from openai import NotGiven, OpenAI, Stream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import ResponseFormat

from python.utils.logger import get_logger

FinishReason: TypeAlias = Literal[
    'stop', 'length', 'tool_calls', 'content_filter', 'function_call'
]

logger = get_logger(__name__)


class ExtraPayloadContents(TypedDict):
    frequency_penalty: float | NotGiven
    max_tokens: int | NotGiven
    presence_penalty: float | NotGiven
    response_format: ResponseFormat | NotGiven
    stop: str | NotGiven
    stream_options: ChatCompletionStreamOptionsParam | NotGiven
    temperature: float | NotGiven
    top_p: int | NotGiven
    tools: Iterable[ChatCompletionToolParam] | NotGiven
    tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven
    logprobs: bool | NotGiven


class LLMProfile:
    """Profile for OpenAI-style API."""

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

    # TODO: Unify hook for both stream and non-stream.
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str,
        stream_hook: CustomStreamHook = no_hook,
        completion_kwargs: ExtraPayloadContents | None = None,
        openai_kwargs: dict[str, Any] | None = None,
        reasoning_field_name: str = 'reasoning_content',
    ) -> None:
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, **(openai_kwargs or {})
        )
        self.model = model
        self.system_prompt = system_prompt
        self.stream_hook = stream_hook
        self.reasoning_field_name = reasoning_field_name

        self.completion_kwargs = (
            completion_kwargs or self._completion_kwargs_default
        )


class LLMWrapper(dict[str, LLMProfile]):
    def __init__(self, profiles: dict[str, LLMProfile] | None = None) -> None:
        self.profiles = profiles or {}

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
        user_prompt: str,
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
        tools: Iterable[ChatCompletionToolParam] | NotGiven | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam
        | NotGiven
        | None = None,
        logprobs: bool | NotGiven | None = None,
    ) -> CustomResponseStr: ...

    @overload
    def completion(
        self,
        profile_key: str,
        user_prompt: str,
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
        tools: Iterable[ChatCompletionToolParam] | NotGiven | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam
        | NotGiven
        | None = None,
        logprobs: bool | NotGiven | None = None,
    ) -> CustomStreamHandler[ChatCompletionChunk]: ...

    def completion(
        self,
        profile_key: str,
        user_prompt: str,
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
        tools: Iterable[ChatCompletionToolParam] | NotGiven | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam
        | NotGiven
        | None = None,
        logprobs: bool | NotGiven | None = None,
    ) -> str | CustomStreamHandler[ChatCompletionChunk]:
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

        completion_kwargs = self._merge_payloads(
            instance_completion_kwargs, profile.completion_kwargs
        )

        logger.debug(
            f'Sent request to {profile.client.base_url} {profile.model}.'
        )
        response = profile.client.chat.completions.create(
            model=profile.model,
            messages=[
                {'role': 'system', 'content': profile.system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            stream=stream,
            **completion_kwargs,
        )

        rfn = profile.reasoning_field_name
        if not stream:
            assert isinstance(response, ChatCompletion)
            cm = response.choices[0].message
            return (
                CustomResponseStr()
                + (
                    ReasoningContentStr()
                    if not hasattr(cm, rfn)
                    else ReasoningContentStr(getattr(cm, rfn))
                )
                + ContentStr(cm.content)
            )

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
                to_str=to_str,
                hook=profile.stream_hook,
                called_by=this_fn,
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

    @staticmethod
    def _merge_payloads(
        fst: ExtraPayloadContents, snd: ExtraPayloadContents
    ) -> ExtraPayloadContents:
        # Use bool(NotGiven()) == False
        return {  # pyright: ignore [reportReturnType]
            k: v_fst or v_snd
            for ((k, v_fst), v_snd) in zip(fst.items(), snd.values())
        }

    @overload
    def _get_completion_function(
        self,
        profile_key: str,
        stream: Literal[False],
        kwargs: ExtraPayloadContents,
    ) -> Callable[[str], CustomResponseStr]: ...

    @overload
    def _get_completion_function(
        self,
        profile_key: str,
        stream: Literal[True],
        kwargs: ExtraPayloadContents,
    ) -> Callable[[str], Generator[ChatCompletionChunk]]: ...

    def _get_completion_function(
        self, profile_key: str, stream: bool, kwargs: ExtraPayloadContents
    ) -> CompletionFn[ChatCompletionChunk]:
        def _completion(
            user_prompt: str,
        ) -> CustomResponseStr | Generator[ChatCompletionChunk]:
            r = self.completion(
                profile_key, user_prompt, stream=stream, **kwargs
            )

            if isinstance(r, CustomResponseStr):
                return r
            else:
                return r._generator

        return _completion
