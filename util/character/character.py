#!/usr/bin/env python3
from dataclasses import dataclass

from character.card import CardManager
from util.ara_id import AraId
from util.bars import BarManager
from util.history import HistoryManager
from util.importance import ImportanceType


@dataclass
class Character:
    char_id: AraId
    card: CardManager
    history: HistoryManager
    bars: BarManager
    capabilities: list[str]
    importance: ImportanceType

    @property
    def ara_id(self) -> AraId:
        return self.char_id
