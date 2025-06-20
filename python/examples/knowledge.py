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
from pprint import pprint as pp

from examples.config import confh
from examples.database import db
from world.knowledge import (
    KAddSpec,
    KSearchSpec,
    KSSExtender,
    RAGKnowledge,
)

world_knowledge = RAGKnowledge('world', db)


def dummy_setup():
    aspec = KAddSpec.new()

    world_knowledge.simple_add(aspec)


# Initialize a dummy database.
if not confh.games.dummy_init:
    dummy_setup()
    confh.games.dummy_init = True  # Flip.
    confh.export_conf3()


def demo_search():
    spec = KSearchSpec.new(
        n_results=5,
        instruct_task='Retrieve passages that directly relate to the query.',
        rerank_task='Given a web search query, retrieve relevant passages that answer the query.',
        with_reranker=True,
        reranker_context=True,
    )

    x = -100

    spec.extend(KSSExtender(0, 'Who was Feanor?', x))
    spec.extend(KSSExtender(1, 'Who were the Teleri?', x))
    spec.extend(KSSExtender(2, 'Where did the elves awaken?', x))
    spec.extend(KSSExtender(3, 'What happened at the Swanhaven?', x))
    r = spec.pop_handle(2)

    pp(world_knowledge[spec])

    spec.extend(r)
    pp(world_knowledge[spec])


if __name__ == '__main__':
    demo_search()
