#!/usr/bin/env python3
# TODO: Add Memory to __init__.py
from typing import Generic, TypeVar
from uuid import UUID
import json

from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)

from llm.database import DatabaseProvider
from llm.api import GameLLM
from llm.utils.stream import CustomHookArgs

T = TypeVar('T')


class Memory(Generic[T]):
    write_scratch = GameLLM.create_tool(
        name='write_scratch',
        description="""Overwrites scratchpad with new transient state.
Use only for:
1) Immediate next actions/scene plans
2) Temporary clues/observations
3) Conversation anchors/observations about emotional state
4) Working theories (not permanent facts).
ALWAYS INCLUDE existing content you want to preserve!
Never store: Historical events, general knowledge, or character backstory.""",
        properties={
            'contents': {
                'type': 'string',
                'description': 'New scratch content. '
                'Replaces all previous content.',
            }
        },
        required=['contents'],
    )

    def __init__(
        self, db: DatabaseProvider | None = None, key: UUID | None = None
    ) -> None:
        self.db: DatabaseProvider | None = db
        self.key: UUID | None = key
        self.scratch: str = ''

        self.write_scratch_hook = GameLLM[T].create_tool_hook(
            name=self.write_scratch['name'],
            tool_hook_contents_fn=self._write_scratch_hook_contents,
        )

    def _write_scratch_hook_contents(
        self,
        _: CustomHookArgs[
            ChatCompletionChunk,
            T,
            ChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ],
        tool_args: str,
    ) -> str:
        self.scratch = json.loads(tool_args)['contents']
        return self.scratch
