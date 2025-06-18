#!/usr/bin/env python3
from typing import (
    Any,
    Callable,
    Generic,
    TypeAlias,
    TypeVar,
    override,
)

from openai.types.shared_params.function_definition import FunctionDefinition

from llm.utils.stream import (
    CustomHookArgs,
    CustomToolHook,
    ToolBlurbStr,
    ToolCallChainExtension,
    ToolCallExtension,
)
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)

from llm.utils.openai_api import LLMProfile, LLMWrapper
from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

CustomToolHookContentsFn: TypeAlias = Callable[
    [
        CustomHookArgs[
            ChatCompletionChunk,
            T,
            ChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ],
        str,
    ],
    str,
]


class GameLLM(LLMWrapper, Generic[T]):
    default_blurb_fstring = '<TOOL//{name}>'

    @override
    def __init__(
        self,
        *profiles: LLMProfile,
        blurb_fstring: str | None = None,
    ) -> None:
        super().__init__(*profiles)
        self.blurb_fstring = blurb_fstring or self.default_blurb_fstring

    @classmethod
    def create_tool(
        cls,
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
    ) -> FunctionDefinition:
        return {
            'name': name,
            'description': description,
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': required,
            },
        }

    @classmethod
    def create_tool_hook(
        cls,
        name: str,
        tool_hook_contents_fn: CustomToolHookContentsFn[T],
        end: bool = False,
    ) -> CustomToolHook[
        ChatCompletionChunk,
        T,
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    ]:
        blurb = ToolBlurbStr(cls.default_blurb_fstring.format(name=name))

        @CustomToolHook
        def custom_tool_hook(
            args: CustomHookArgs[
                ChatCompletionChunk,
                T,
                ChatCompletionMessage,
                ChatCompletionMessageToolCall,
            ],
        ) -> (
            ToolCallExtension
            | ToolCallChainExtension[ChatCompletionChunk]
            | None
        ):
            cm = args.context_manager

            tools = tuple(
                tool
                for tool in (args.tools or [])
                if tool.function.name == name
            )

            if not tools:
                return None

            assert len(tools) == 1
            tool = tools[0]

            logger.debug(f'Used tool {name}.')

            content = tool_hook_contents_fn(args, tool.function.arguments)

            if end:
                return None

            cm.assistant_message(
                args.text.content_with_reasoning,
                [tool],
                suppress_decorations=True,
            )
            cm.tool_message(content=content, tool_call_id=tool.id)

            iter = args.called_by(cm, None)

            # XXX: This is nessecary
            if isinstance(iter, str):
                return blurb, iter
            else:
                return blurb, iter

        return custom_tool_hook
