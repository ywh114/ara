from examples.config import confh
from examples.database import db
from world.knowledge import (
    KAddSpec,
    KSearchSpec,
    KSSExtender,
    KASExtender,
    RAGKnowledge,
)
from pprint import pprint as pp

world_knowledge = RAGKnowledge('world', db)


def dummy_setup():
    aspec = KAddSpec.new()
    aspec.extend(KASExtender(0, 'Feanor was an elf', {'entity': True}, 'id0'))
    aspec.extend(
        KASExtender(1, 'Feanor did nothing wrong', {'entity': True}, 'id1')
    )
    aspec.extend(
        KASExtender(2, 'Alqualonde is the Swanhaven', {'location': True}, 'id2')
    )
    aspec.extend(
        KASExtender(3, 'Alqualonde is by the sea', {'location': True}, 'id3')
    )
    aspec.extend(
        KASExtender(4, 'Cuivienon is a lake', {'location': True}, 'id4')
    )
    aspec.extend(
        KASExtender(5, 'Elves are the Firstborn', {'entity': True}, 'id5')
    )
    aspec.extend(
        KASExtender(
            6,
            'Elves awakened at Cuivienon',
            {'location': True, 'entity': True},
            'id6',
        )
    )
    aspec.extend(
        KASExtender(
            7,
            'Feanor killed kin at Alqualonde',
            {'location': True, 'entity': True},
            'id7',
        )
    )
    aspec.extend(
        KASExtender(8, 'Feanor was of the Noldor', {'location': True}, 'id8')
    )
    aspec.extend(
        KASExtender(9, 'Noldor are a clan of elves', {'entity': True}, 'id9')
    )
    aspec.extend(
        KASExtender(10, 'Teleri are a clan of elves', {'entity': True}, 'id10')
    )
    aspec.extend(
        KASExtender(
            11,
            'The Teleri lived at Alqualonde',
            {'entity': True, 'location': True},
            'id11',
        )
    )
    aspec.extend(
        KASExtender(12, 'The Teleri had it coming', {'entity': True}, 'id12')
    )
    aspec.extend(
        KASExtender(
            13, 'The Teleri and Noldor are kin', {'entity': True}, 'id13'
        )
    )

    world_knowledge.simple_add(aspec)


# Initialize a dummy database.
if not confh.games.dummy_init:
    dummy_setup()
    confh.games.dummy_init = True
    confh.export_conf3()

spec = KSearchSpec.new(
    n_results=5,
    instruct_task='Retrieve passages that directly relate to the query.',
    rerank_task='Given a web search query, retrieve relevant passages that answer the query.',
    with_reranker=False,
)

x = 0.5

spec.extend(KSSExtender(0, 'Who was Feanor?', x))
spec.extend(KSSExtender(1, 'Who were the Teleri?', x))
spec.extend(KSSExtender(2, 'Where did the elves awaken?', x))
spec.extend(KSSExtender(3, 'What happened at the Swanhaven?', x))
r = spec.pop_handle(2)

pp(world_knowledge[spec])

spec.extend(r)
pp(world_knowledge[spec])
