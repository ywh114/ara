#!/usr/bin/env python3
from uuid import uuid4
from utils.bars import BarManager
from world.character.character_class import Character
from world.character.memory import Memory
from examples.card import cardh
from examples.importance import CharacterImportance

char = Character(
    uuid4(), cardh, Memory(), BarManager(), [], CharacterImportance.REQUIRED
)
