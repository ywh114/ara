#!/usr/bin/env python3
from dataclasses import dataclass
from uuid import UUID

from character.card import CardHolder
from character.memory import Memory
from util.bars import BarManager
from world.importance import ImportanceType


@dataclass
class Character:
    id: UUID
    card: CardHolder
    memory: Memory
    bars: BarManager
    capabilities: list[str]
    importance: ImportanceType
