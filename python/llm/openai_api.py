#!/usr/bin/env python3
import os
from typing import (
    Any,
    Iterable,
    Iterator,
    Literal,
    TypeVar,
    TypedDict,
    overload,
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

sk = os.environ[
    'DEEPSEEK_API_KEY'
]  # TODO: Add environment variable name to settings.

print(sk)


T = TypeVar('T')


class CustomStreamHandler(Iterator[T]):
    def __init__(self, stream: Stream[T]) -> None:
        self.stream = stream
        self._iterator = (chunk for chunk in stream)

    def __next__(self) -> T:
        return next(self._iterator)


class ExtraPayloadContents(TypedDict):
    # NOTE: Does not include `stream`.
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


class LLM:
    """Wrapper for OpenAI API."""

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
        openai_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, **(openai_kwargs or {})
        )
        self.model = model
        self._completion_kwargs: ExtraPayloadContents = (
            self._completion_kwargs_default.copy()
        )

    @overload
    def completion(self, *args, stream: Literal[False], **kwargs) -> str: ...

    @overload
    def completion(
        self, *args, stream: Literal[True], **kwargs
    ) -> Stream[ChatCompletionChunk]: ...

    def completion(
        self,
        system_prompt: str,
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
    ) -> str | Stream[ChatCompletionChunk]:
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
                {'role': 'system', 'content': system_prompt},
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
            return response


llm = LLM(sk, 'https://api.deepseek.com', model='deepseek-chat')

r = llm.completion(
    'You are a helpful assistant',
    'Hello',
    stream=True,
)

for chunk in r:
    print(chunk.choices)

print(r.response)
