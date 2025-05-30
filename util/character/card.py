#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class Card:
    """A character card for in-character LLM response generation."""

    name: str | None
    physical_traits: str | None
    personality_traits: str | None
    scenario: str | None
    examples: str | None
    voice: str | None


@dataclass
class CardManager:
    card: Card
