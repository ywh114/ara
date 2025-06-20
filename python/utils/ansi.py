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
# ANSI escape sequences
__all__ = [
    'BLACK',
    'DARKGRAY',
    'DARKGREY',
    'DARKRED',
    'RED',
    'DARKGREEN',
    'GREEN',
    'DARKYELLOW',
    'YELLOW',
    'DARKBLUE',
    'BLUE',
    'DARKMAGENTA',
    'MAGENTA',
    'DARKCYAN',
    'CYAN',
    'LIGHTGRAY',
    'LIGHTGREY',
    'WHITE',
    'UNDERLINE',
    'NOUNDERLINE',
    'BOLD',
    'NOBOLD',
    'RESET',
    'CLEAR',
    'END',
]

BLACK = '\033[30m'
DARKGRAY = DARKGREY = '\033[90m'

DARKRED = '\033[31m'
RED = '\033[91m'

DARKGREEN = '\033[32m'
GREEN = '\033[92m'

DARKYELLOW = '\033[33m'
YELLOW = '\033[93m'

DARKBLUE = '\033[34m'
BLUE = '\033[94m'

DARKMAGENTA = '\033[35m'
MAGENTA = '\033[95m'

DARKCYAN = '\033[36m'
CYAN = '\033[96m'

LIGHTGRAY = LIGHTGREY = '\033[37m'
WHITE = '\033[97m'

UNDERLINE = '\033[4m'
NOUNDERLINE = '\033[24m'
BOLD = '\033[1m'
NOBOLD = '\033[21m'

RESET = '\033[39m\033[49m'
CLEAR = '\033[2K'

END = '\033[0m'
