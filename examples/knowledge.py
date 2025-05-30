from util.knowledge import Knowledge  # , GatedKnowledge


class WorldKnowledge(Knowledge):
    domain_name = 'World knowledge'

    @classmethod
    def retrieve(cls) -> str:
        return 'hi'


print(WorldKnowledge.domain_name)
print(WorldKnowledge.retrieve())
