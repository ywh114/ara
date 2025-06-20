#!/usr/bin/env python3
from pathlib import Path

from examples.config import confh, example_assets
from utils.logger import get_logger
from world.character.card import standard_load

logger = get_logger(__name__)

example_card0 = Path('柴郡.png')
example_card1 = Path('香槟.png')
example_card2 = Path('Player.png')
example_card3 = Path('明石.png')

cardh0 = standard_load(example_card0, example_assets, confh)
cardh1 = standard_load(example_card1, example_assets, confh)
cardh2 = standard_load(example_card2, example_assets, confh)
cardh3 = standard_load(example_card3, example_assets, confh)


if __name__ == '__main__':
    logger.info(cardh0)
    logger.info(cardh1)
    logger.info(cardh2)
    logger.info(cardh3)
