#!/usr/bin/env python3
import random
from typing import Hashable
from uuid import UUID


def uuid4_from_seed(seed: Hashable) -> UUID:
    random.seed(hash(seed))
    random_bytes = bytes([random.getrandbits(8) for _ in range(16)])
    return UUID(bytes=random_bytes, version=4)
