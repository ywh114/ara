#!/usr/bin/env python3
from examples.config import confh
from llm.database import Chroma

from util.logger import get_logger


logger = get_logger(__name__)

db = Chroma(confh)
