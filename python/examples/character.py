#!/usr/bin/env python3
import random
import uuid

from examples.card import cardh
from examples.database import db
from examples.importance import CharacterImportance
from utils.bars import BarManager
from world.character.character_class import Character
from world.character.memory import Memory


def uuid4_from_seed(seed: int) -> uuid.UUID:
    random.seed(seed)
    random_bytes = bytes([random.getrandbits(8) for _ in range(16)])
    return uuid.UUID(bytes=random_bytes, version=4)


seed = 114514

char = Character(
    char_id := uuid4_from_seed(seed),
    cardh,
    Memory(db, char_id),
    BarManager(),
    [],
    CharacterImportance.REQUIRED,
)

if __name__ == '__main__':
    print(char.id)
