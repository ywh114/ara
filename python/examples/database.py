#!/usr/bin/env python3
from examples.config import confh
from llm.database import Chroma

from utils.logger import get_logger


logger = get_logger(__name__)

db = Chroma(confh)

if __name__ == '__main__':
    logger.info(db)
