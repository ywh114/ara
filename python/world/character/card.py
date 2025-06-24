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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias
from uuid import UUID

import aichar
from configuration.config import (
    P,
    Q,
    ConfigHolder,
    GameConfig,
    InstallationConfig,
)
from configuration.structure import (
    V,
    X,
    GameSettings,
    InstallationSettings,
)
from utils.logger import get_logger
from utils.uuid4_from_seed import uuid4_from_seed

logger = get_logger(__name__)


CardFormat: TypeAlias = (
    Literal['tavernai']
    | Literal['sillytavern']
    | Literal['textgenerationwebui']
    | Literal['pygmalion']
    | Literal['aicompanion']
)

default_format: CardFormat = 'sillytavern'


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

    :param id: Unique UUID identifier for the card.
    :param path: Filesystem path to the card image.
    :param cc: Loaded :class:`aichar.CharacterClass` instance.
    :param export_format: Export format (default: :attr:`CardFormat.sillytavern`).

    :raises ValueError: If path doesn't match CharacterClass's image_path.
    """

    id: UUID
    path: Path
    cc: Any  # `CharacterClass` is not exposed by the library.
    export_format: CardFormat = default_format

    def __post_init__(self) -> None:
        """
        Validate path consistency after initialization.

        :raises ValueError: When path doesn't match CharacterClass's
            `image_path`.
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
        export_format: CardFormat = default_format,
    ) -> Self:
        """
        Load character card from PNG with optional format conversion.

        :param card_id: The unique identifier for the card.
        :param path: Target path for the card file.
        :param init_path: Source path for conversion (uses direct load if None).
        :param export_format: Target export format.
        :return: Initialized CardHolder instance,

        When `init_path` is provided:
          If `path` does not exist:
            1. Loads card from `init_path`.
            2. Exports to `path` in specified format.
            3. Updates internal path reference.
        """
        if init_path is None or path.is_file():
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
        :param value: New field value.
        """
        setattr(self.cc, field, value)

    def get_field(self, field: CardField) -> str:
        """
        Retrieve character card field value.

        :param field: Field to access.
        :return: Current field value.
        """
        return getattr(self.cc, field)

    def __repr__(self) -> str:
        return self.cc.data_summary


def standard_load(
    path: Path,
    confh: ConfigHolder[P, Q, V, X],
) -> CardHolder:
    # TODO: Put assets_dir into confh.insts
    return CardHolder.load_png(
        uuid4_from_seed(path.stem),
        path=confh.insts.cache_session_cc_dir.joinpath(path),
        init_path=confh.insts.cache_assets_cc_dir.joinpath(path),
    )
