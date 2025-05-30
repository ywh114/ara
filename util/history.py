#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class History:
    pass


@dataclass
class HistoryManager:
    history: History
    pass
