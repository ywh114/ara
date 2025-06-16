#!/usr/bin/env python3
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias
from uuid import UUID

import aichar
from utils.logger import get_logger

logger = get_logger(__name__)


class CardFormat(StrEnum):
    """
    Supported character card export formats.

    :cvar tavernai: TavernAI character card format
    :cvar sillytavern: SillyTavern character card format (default)
    :cvar textgenerationwebui: Text Generation WebUI character card format
    :cvar pygmalion: Pygmalion character card format
    :cvar aicompanion: AI Companion character card format
    :cvar default: Alias for the default format (sillytavern)
    """

    TAVERNAI = auto()
    SILLYTAVERN = auto()
    TEXTGENERATIONWEBUI = auto()
    PYGMALION = auto()
    AICOMPANION = auto()

    DEFAULT = SILLYTAVERN


CardField: TypeAlias = (
    Literal['name']
    | Literal['summary']
    | Literal['personality']
    | Literal['scenario']
    | Literal['greeting_message']
    | Literal['example_messages']
)


@dataclass
class CardHolder:
    """
    Holds character card data with conversion and manipulation capabilities.

    :param id: Unique AraId identifier for the card
    :param path: Filesystem path to the card image
    :param cc: Loaded :class:`aichar.CharacterClass` instance
    :param export_format: Export format (default: :attr:`CardFormat.sillytavern`)

    :raises ValueError: If path doesn't match CharacterClass's image_path
    """

    id: UUID
    path: Path
    cc: Any  # `CharacterClass` is not exposed by the library.
    export_format: CardFormat = CardFormat.DEFAULT

    def __post_init__(self) -> None:
        """
        Validate path consistency after initialization.

        :raises ValueError: When path doesn't match CharacterClass's image_path
        """
        if not self.cc.image_path != self.path:
            raise ValueError(
                '`path`* must always match `card.image_path`**, but '
                f'`{self.path}`* is not (2)`{self.cc.image_path}`**.'
            )

    @classmethod
    def load_png(
        cls,
        _id: UUID,
        path: Path,
        init_path: Path | None = None,
        export_format: CardFormat = CardFormat.DEFAULT,
    ) -> Self:
        """
        Load character card from PNG with optional format conversion.

        :param card_id: The unique identifier for the card.
        :type card_id: AraId
        :param path: Target path for the card file.
        :type path: Path
        :param init_path: Source path for conversion (uses direct load if None).
        :type init_path: Path
        :param export_format: Target export format.
        :type export_format: CardFormat
        :return: Initialized CardHolder instance,
        :rtype: Self

        When `init_path` is provided:
          1. Loads card from `init_path`.
          2. Exports to `path` in specified format.
          3. Updates internal path reference.
        """
        if init_path is None:
            logger.debug(f'Load {cls.__name__} from {path}')
            return cls(
                _id,
                path,
                aichar.load_character_card_file(path.as_posix()),
                export_format,
            )
        else:
            # 1. Load card from `init_path`.
            logger.debug(f'Load {cls.__name__} from {init_path}')
            cc = aichar.load_character_card_file(init_path.as_posix())
            # 2. Write card to `path`.
            logger.debug(
                f'Change path and export {cls.__name__} to {init_path}'
            )
            path.parents[0].mkdir(parents=True, exist_ok=True)  # Mkdir parents.
            cc.export_card_file(
                cls.export_format if export_format is None else export_format,
                path.as_posix(),
            )
            # 3. Change `card`[init_path] to `card`[path].
            cc.image_path = path.as_posix()
            # Return cls(path, card[path]).
            return cls(_id, path, cc, export_format)

    def export_png(self) -> None:
        """Export character card to PNG using configured format."""
        logger.debug(f'Export {self.__class__.__name__} to {self.path}')
        self.cc.export_card_file(self.export_format, self.path.as_posix())

    def set_field(self, field: CardField, value: str) -> None:
        """
        Update character card field value.

        :param field: Field to modify.
        :type field: CardField
        :param value: New field value.
        :type value: str
        """
        setattr(self.cc, field, value)

    def get_field(self, field: CardField) -> str:
        """
        Retrieve character card field value.

        :param field: Field to access.
        :type field: CardField
        :return: Current field value.
        :rtype: str
        """
        return getattr(self.cc, field)

    def __repr__(self) -> str:
        return self.cc.data_summary
