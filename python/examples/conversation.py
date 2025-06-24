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
from pprint import pp

from examples.character import narrator, player
from examples.config import confh
from utils.logger import get_logger
from world.character_roles import RoleProfiles
from examples.plot import pm
from world.conversation import RPInfo, multiround_conversation


logger = get_logger(__name__)

rp = RoleProfiles(confh)


def user_prompt_from_input(info: RPInfo) -> str:
    print('\n'.join(info.suggestions))
    user_input = input('User> ').lower().strip()

    if user_input.startswith(('/', ':')):
        if __debug__:
            info.get_user_debug(info, user_input[1:])
        else:
            logger.critical(f'Failure - `__debug__`: {__debug__}.')
        return info.get_user_prompt(info)

    return user_input


def user_debug_from_input(info: RPInfo, noshell: str = '') -> None:
    def handle(com: list[str]) -> bool:
        if len(com) == 0:
            return False
        elif com[0] in ('help', 'h'):
            print("""
:commands (! -> to be implemented)
        -> continue
help, h -> this message
dump, d -> dump context
info, i -> dump info
a       -> dir(info)
ga *... -> getattr(.) for . in *...
n       -> dump next character
pl      -> dump player
na      -> dump narrator
here    -> dump here characters
away    -> dump away characters
div     -> dump next directive
sug     -> dump next suggestions
en      -> dump entering characters     !
ex      -> dump exiting characters      !
pool    -> dump character pool          !
exec, x ...
        -> exec(...)    [debug console only]
debug   -> enter debug  [user prompt only]
exit    -> ^C
""")
        elif com[0] in ('dump', 'd'):
            pp(info.cm.context)
        elif com[0] in ('info', 'i'):
            pp(info)
        elif com[0] in ('a',):
            pp(dir(info))
        elif com[0] in ('ga',):
            if len(com) > 1:
                for arg in com[1:]:
                    try:
                        pp(getattr(info, arg))
                    except AttributeError:
                        logger.warning(arg + 'not found.')
        elif com[0] in ('n',):
            pp(info.next_char)
        elif com[0] in ('pl',):
            pp(info.player_char)
        elif com[0] in ('na',):
            pp(info.narrator_char)
        elif com[0] in ('here',):
            pp(info.here_chars)
        elif com[0] in ('away',):
            pp(info.away_chars)
        elif com[0] in ('div',):
            pp(info.directive)
        elif com[0] in ('sug',):
            pp(info.suggestions)
        elif com[0] in ('exec', 'x'):
            if noshell:
                logger.critical('Disabled in prompt mode.')
            elif len(com) > 1:
                try:
                    exec(' '.join(com[1:]))
                except Exception as e:
                    logger.warning(e)
        elif com[0] in ('debug',):
            if noshell:
                info.get_user_debug(info, '')
        elif com[0] in ('exit',):
            raise KeyboardInterrupt
        else:
            handle(['h'])

        return True

    if noshell:
        handle(noshell.split())
    else:
        handle(['h'])
        while handle(input('Debug> :').lower().strip().split()):
            pass


try:
    multiround_conversation(
        rp,
        pm,
        player_char=player,
        narrator_char=narrator,
        get_user_prompt=user_prompt_from_input,
        get_user_debug=user_debug_from_input,
    )
except KeyboardInterrupt:
    logger.critical('Caught keyboard interrupt.')
