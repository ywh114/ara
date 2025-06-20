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
# TODO: Add Memory to __init__.py
import json
from typing import Generic, TypeVar
from uuid import UUID

from llm.api import GameLLM
from llm.database import DatabaseProvider
from llm.utils.openai_api import ToolsHookPair
from llm.utils.stream import CustomHookArgs
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from utils.logger import get_logger
from world.knowledge import (
    KSearchSpec,
    KSearchSpecExtender,
    RAGKnowledge,
)

logger = get_logger(__name__)


T = TypeVar('T')


write_scratch = GameLLM.create_tool(
    name='write_scratch',
    description="""Overwrites scratchpad.
Use for:
1) Next actions/scene plans (i.e. if you decide to lie, don't forget you were lying later on)
2) Clues/observations (store things that may be relevant in the future here)
3) Conversation anchors/observations about characters' emotional states (same as above)
4) Working theories, not permanent facts (don't speak all your theories out loud)
5) Important tool call results (that you aren't going to immediately speak out loud)
ALWAYS INCLUDE EXISTING CONTENT YOU WANT TO PRESERVE!
If there is nothing you want to change, do not provide the `contents` field.""",
    properties={
        'contents': {
            'type': 'string',
            'description': 'New scratch content. '
            'Replaces all previous content.',
        }
    },
    required=[],
)
recall_from_conversations = GameLLM.create_tool(
    name='recall_conversation',
    description="""Attempt to recall snippets from prior conversations.""",
    properties={
        'query': {
            'type': 'array',
            'items': {'type': 'string'},
            'description': 'List of query/search texts.\n',
        },
        'strength': {
            'type': 'string',
            'description': 'How many results to return. '
            'One of "shallow", "medium", "deep", "very_deep".',
        },
    },
    required=['query', 'strength'],
)


class Memory(Generic[T]):
    default_scratch = 'Nothing yet!'

    def __init__(
        self, db: DatabaseProvider | None = None, key: UUID | None = None
    ) -> None:
        self.db: DatabaseProvider | None = db
        self.key: UUID | None = key
        self.scratch: str = self.default_scratch
        self.prev_scratch: str = self.default_scratch

        self.conversation_knowledge = None
        if self.db is not None and self.key is not None:
            self.conversation_knowledge = RAGKnowledge(str(self.key), self.db)

        self.write_scratch_hook = GameLLM.create_tool_hook(
            name=write_scratch['name'],
            tool_hook_contents_fn=self._write_scratch_hook_contents,
        )
        self.write_scratch_hook_end = GameLLM.create_tool_hook(
            name=write_scratch['name'],
            tool_hook_contents_fn=self._write_scratch_hook_contents,
            end=True,
        )
        self.recall_from_conversations_hook = GameLLM.create_tool_hook(
            name=recall_from_conversations['name'],
            tool_hook_contents_fn=self._recall_from_conversations_hook_contents,
        )

        self.chat_tools = ToolsHookPair(
            tools=[recall_from_conversations],
            hook=self.recall_from_conversations_hook,
        )
        self.chat_tools_end = ToolsHookPair(
            tools=[recall_from_conversations, write_scratch],
            hook=self.recall_from_conversations_hook
            | self.write_scratch_hook_end,
        )

    def prepare_scratch_for_new_conversation(self) -> None:
        self.prev_scratch = self.scratch
        self.scratch = self.default_scratch

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
        args_dict = json.loads(tool_args)
        if 'contents' in args_dict:
            self.scratch = args_dict['contents']
            return 'New contents:\n' + self.scratch
        else:
            return 'No changes made.'

    def _recall_from_conversations_hook_contents(
        self,
        _: CustomHookArgs[
            ChatCompletionChunk,
            T,
            ChatCompletionMessage,
            ChatCompletionMessageToolCall,
        ],
        tool_args: str,
    ) -> str:
        if self.conversation_knowledge is None:
            return 'Character does not support conversation history. '
        "Admit you don't know, or make something up depending on the "
        'situation. You probably want to do the prior.'

        args_dict = json.loads(tool_args)
        queries: list[str] = args_dict['query']
        strength = args_dict['strength']

        n_results, rerank = {
            'shallow': (2, False),
            'medium': (5, True),
            'deep': (10, True),
            'very_deep': (30, False),
        }[strength]

        mm_dist = (2.5, 0.2)[rerank]

        spec = KSearchSpec.new(
            n_results, with_reranker=rerank, where={'memory': {'$eq': True}}
        )
        for query in queries:
            spec.extend(KSearchSpecExtender(0, query, mm_dist))

        results = self.conversation_knowledge[spec]

        ret = ''
        for inst, qr in zip(results.instructions, results.query_results):
            ret += f'{inst}:\n'
            if (docs := qr['documents']) is None:
                return 'Not found.'
            for doc in docs:
                ret += f'```snippet\n{doc}\n```\n'

        return 'Not found.'
