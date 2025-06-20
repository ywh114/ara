#!/usr/bin/env python3
from examples.card import cardh0, cardh1, cardh2, cardh3
from examples.database import db
from examples.importance import CharacterImportance
from utils.bars import BarManager
from utils.logger import get_logger
from utils.uuid4_from_seed import uuid4_from_seed
from world.character.character_class import Character
from world.character.memory import Memory

logger = get_logger(__name__)

seed0 = 114514
seed1 = 191981
seed2 = 666666
seed3 = 999999

char0 = Character(
    char_id := uuid4_from_seed(seed0),
    cardh0,
    Memory(db, char_id),
    BarManager(),
    [],
    CharacterImportance.REQUIRED,
)

char1 = Character(
    char_id := uuid4_from_seed(seed1),
    cardh1,
    Memory(db, char_id),
    BarManager(),
    [],
    CharacterImportance.REQUIRED,
)

player = Character(
    char_id := uuid4_from_seed(seed2),
    cardh2,
    Memory(db, char_id),
    BarManager(),
    [],
    CharacterImportance.EIGEN,
)

narrator = Character(
    char_id := uuid4_from_seed(seed3),
    cardh3,
    Memory(db, char_id),
    BarManager(),
    [],
    CharacterImportance.REQUIRED,
)

if __name__ == '__main__':
    logger.info(char0)
    logger.info(char1)
    logger.info(player)
    logger.info(narrator)
