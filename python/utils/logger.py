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
import logging
from typing import TypeAlias

from utils.ansi import BOLD, END, GREEN, LIGHTGREY, RED, YELLOW

_Level: TypeAlias = str | int


class ColorFormatter(logging.Formatter):
    format0 = '%(levelname)s - %(message)s (%(filename)s:%(lineno)d)'
    format1 = '%(asctime)s - %(name)s: %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'

    FORMATS = {
        logging.DEBUG: LIGHTGREY + format1 + END,
        logging.INFO: GREEN + format0 + END,
        logging.WARNING: YELLOW + format1 + END,
        logging.ERROR: RED + format1 + END,
        logging.CRITICAL: BOLD + RED + format1 + END,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str, level: _Level | None = None) -> logging.Logger:
    ch = logging.StreamHandler()
    ch.setFormatter(ColorFormatter())

    logger = logging.getLogger(name)
    logger.addHandler(ch)
    if level is None:
        if __debug__:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    else:
        logger.setLevel(level)

    return logger
