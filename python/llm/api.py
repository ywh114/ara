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
from typing import (
    Any,
    Callable,
    TypeAlias,
    TypeVar,
    override,
)

from llm.utils.openai_api import LLMProfile, LLMWrapper
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
from openai.types.shared_params.function_definition import FunctionDefinition
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


class GameLLM(LLMWrapper[T]):
    """
    Specialized LLM wrapper for game environments with tool integration support.

    :param profiles: LLM profile configurations for API interactions.
    :param blurb_fstring: Format string for tool blurbs.
    """

    default_blurb_fstring = '<used tool {name}>'
    """Default fstring for tool blurbs."""

    @override
    def __init__(
        self,
        *profiles: LLMProfile[
            ChatCompletionChunk,
            T,
            ChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ],
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
        """
        Generate a tool definition for LLM function calling.

        :param name: Unique tool identifier.
        :param description: Natural language description of tool purpose.
        :param properties: Schema defining tool parameters.
        :param required: Mandatory parameters from properties.
        :return: Structured tool definition compatible with OpenAI API.
        """
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
    def vectorize_tools(cls, *tools: FunctionDefinition) -> FunctionDefinition:
        raise NotImplementedError

    @classmethod
    def vectorize_hook(
        cls,
        *tools: CustomToolHook[
            ChatCompletionChunk,
            T,
            ChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ],
    ) -> CustomToolHook[
        ChatCompletionChunk,
        T,
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    ]:
        raise NotImplementedError

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
        """
        Create a hook handler for processing tool calls during streaming.

        :param name: Target tool name to handle.
        :param tool_hook_contents_fn: Hook that processes tool arguments and
        returns content.
        :param end: Flag indicating terminal tool in processing chain.
        :return: Configured tool hook for integration with streaming pipeline.
        """
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

            # Padding.
            if cm.head is not None and cm.head['role'] == 'assistant':
                cm.user_message('', suppress_decorations=True)

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
