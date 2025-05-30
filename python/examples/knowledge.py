from datetime import datetime
from llm.database import Database
from examples.config import confh
from llm.database import Chroma
from world.knowledge import KSearchSpec, KSSExtender, RAGKnowledge
from pprint import pprint as pp

dt = datetime.now()

db = Database(Chroma, confh)
world_knowledge = RAGKnowledge('world', db)

spec = KSearchSpec.new()

spec.extend(KSSExtender(0, 'Who was Feanor', 5, 1))
# spec.extend(KSSExtender(1, 'Where are the lakes', 3, 1))
# spec.extend(KSSExtender(2, 'Where did the elves awaken', 5, 1))

pp(dt.now())
pp(world_knowledge[spec])
pp(dt.now())
