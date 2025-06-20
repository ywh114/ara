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
