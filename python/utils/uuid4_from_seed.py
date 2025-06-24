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
import random
from typing import Hashable
from uuid import UUID


def uuid4_from_seed(seed: Hashable) -> UUID:
    """
    Generate a deterministic UUIDv4 from a seed value.

    :param seed: A hashable value used to seed the random number generator
    :return: A deterministic UUIDv4 generated from the seed
    """
    random.seed(hash(seed))
    random_bytes = bytes(random.getrandbits(8) for _ in range(16))
    return UUID(bytes=random_bytes, version=4)
