#!/usr/bin/env python3
from uuid import uuid4

from util.logger import get_logger
from world.character.card import CardHolder

from examples.config import confh, example_assets

logger = get_logger(__name__)

example_card = 'example_card.png'
cardh = CardHolder.load_png(
    uuid4(),
    path=confh.insts.cache_assets_dir.joinpath(example_card),
    init_path=example_assets.joinpath(example_card),
)


if __name__ == '__main__':
    logger.info(cardh)
