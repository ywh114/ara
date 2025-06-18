#!/usr/bin/env python3
# TODO: Add Memory to __init__.py
from typing import Generic, Literal, TypeAlias, TypeVar
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
from world.knowledge import (
    KSearchSpec,
    KSearchSpecExtender,
    RAGKnowledge,
)

T = TypeVar('T')

RecallStrength: TypeAlias = (
    Literal['shallow'] | Literal['medium'] | Literal['deep']
)


class Memory(Generic[T]):
    write_scratch = GameLLM.create_tool(
        name='write_scratch',
        description="""Overwrites scratchpad.
Use for:
1) Next actions/scene plans (i.e. if you decide to lie, don't forget you were lying later on)
2) Clues/observations (store things that may be relevant in the future here)
3) Conversation anchors/observations about characters' emotional states (same as above)
4) Working theories, not permanent facts (don't speak all your theories out loud)
5) Important tool call results (that you aren't going to immediately speak out loud)
ALWAYS INCLUDE EXISTING CONTENT YOU WANT TO PRESERVE!""",
        properties={
            'contents': {
                'type': 'string',
                'description': 'New scratch content. '
                'Replaces all previous content.',
            }
        },
        required=['contents'],
    )
    recall_from_conversations = GameLLM.create_tool(
        name='recall_conversation',
        description="""Attempt to recall snippets from prior conversations.""",
        properties={
            'query': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Colon-separated list of query texts/'
                'similarity search texts.\n'
                'i.e. "query1:query2:..."',
            },
            'strength': {
                'type': RecallStrength,
                'description': 'How many results to return. '
                'Fastest/faster/slower/slowest.',
            },
        },
        required=['query', 'strength'],
    )

    def __init__(
        self, db: DatabaseProvider | None = None, key: UUID | None = None
    ) -> None:
        self.db: DatabaseProvider | None = db
        self.key: UUID | None = key
        self.scratch: str = 'Nothing yet!'

        self.conversation_knowledge = None
        if self.db is not None and self.key is not None:
            self.conversation_knowledge = RAGKnowledge(str(self.key), self.db)

        self.write_scratch_hook = GameLLM[T].create_tool_hook(
            name=self.write_scratch['name'],
            tool_hook_contents_fn=self._write_scratch_hook_contents,
        )
        self.write_scratch_hook_end = GameLLM[T].create_tool_hook(
            name=self.write_scratch['name'],
            tool_hook_contents_fn=self._write_scratch_hook_contents,
            end=True,
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
        'situation. You probably want the prior.'

        args_dict = json.loads(tool_args)
        queries: list[str] = args_dict['query'].split(':')
        strength: RecallStrength = args_dict['strength']

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
