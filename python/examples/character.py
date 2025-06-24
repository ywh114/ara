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
from examples.database import db
from examples.config import confh
from utils.logger import get_logger
from world.character.character_class import standard_load
# from utils.bars import BarManager

logger = get_logger(__name__)

char0 = standard_load('柴郡', db=db, confh=confh)
char1 = standard_load('香槟', db=db, confh=confh)
player = standard_load('Player', db=db, confh=confh)
narrator = standard_load('明石', db=db, confh=confh)


if __name__ == '__main__':
    logger.info(char0)
    logger.info(char1)
    logger.info(player)
    logger.info(narrator)
