#!/usr/bin/env python3
# TODO: Implement profiles.
# TODO: Add environment variable name to settings.
from typing import (
    Any,
    Iterable,
    Literal,
    TypeAlias,
    TypedDict,
    overload,
)

from llm.utils.stream import (
    ContentStr,
    CustomChunkToStr,
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

FinishReason: TypeAlias = Literal[
    'stop', 'length', 'tool_calls', 'content_filter', 'function_call'
]


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


class APIWrapper:
    """Wrapper for OpenAI-style API."""

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

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str,
        stream_hook: CustomStreamHook = no_hook,
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

        self._completion_kwargs: ExtraPayloadContents = (
            self._completion_kwargs_default.copy()
        )

    @overload
    def completion(self, *args, stream: Literal[False], **kwargs) -> str: ...

    @overload
    def completion(
        self, *args, stream: Literal[True], **kwargs
    ) -> CustomStreamHandler[ChatCompletionChunk]: ...

    def completion(
        self,
        user_prompt: str,
        reuse_options: bool = False,
        *,
        stream: bool = False,
        frequency_penalty: float | None = None,
        max_tokens: int | None = None,
        presence_penalty: float | None = None,
        response_format: ResponseFormat | None = None,
        stop: str | None = None,
        stream_options: ChatCompletionStreamOptionsParam | None = None,
        temperature: float | None = None,
        top_p: int | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
        logprobs: bool | None = None,
    ) -> str | CustomStreamHandler[ChatCompletionChunk]:
        if not reuse_options:  # Ignore without warning.
            self._completion_kwargs = {
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            stream=stream,
            **self._completion_kwargs,
        )

        if not stream:
            assert isinstance(response, ChatCompletion)
            return response.choices[0].message.content or ''
        else:
            assert isinstance(response, Stream)
            return CustomStreamHandler(
                response,
                to_str=CustomChunkToStr(self._chunk_to_str),
                hook=self.stream_hook,
            )

    def _chunk_to_str(self, chunk: ChatCompletionChunk) -> DeltaStr:
        cd = chunk.choices[0].delta

        # For reasoning models.
        if hasattr(cd, self.reasoning_field_name):
            if (c := getattr(cd, self.reasoning_field_name)) is not None:
                assert isinstance(c, str)
                return ReasoningContentStr(c)

        if cd.content is not None:
            return ContentStr(cd.content)
        else:
            return ContentStr()
