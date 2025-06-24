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
from pathlib import Path

from examples.config import confh
from utils.logger import get_logger
from world.character.card import standard_load

logger = get_logger(__name__)


def example_cards() -> None:
    example_card0 = Path('柴郡/柴郡.png')
    example_card1 = Path('香槟/香槟.png')
    example_card2 = Path('Player/Player.png')
    example_card3 = Path('明石/明石.png')

    cardh0 = standard_load(example_card0, confh)
    cardh1 = standard_load(example_card1, confh)
    cardh2 = standard_load(example_card2, confh)
    cardh3 = standard_load(example_card3, confh)

    logger.info(cardh0)
    logger.info(cardh1)
    logger.info(cardh2)
    logger.info(cardh3)


if __name__ == '__main__':
    example_cards()
