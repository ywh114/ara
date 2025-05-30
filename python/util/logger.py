#!/usr/bin/env python3
import logging
from typing import TypeAlias
from util.ansi import GREEN, LIGHTGREY, YELLOW, RED, BOLD, RESET

_Level: TypeAlias = str | int


class ColorFormatter(logging.Formatter):
    format0 = '%(levelname)s - %(message)s (%(filename)s:%(lineno)d)'
    format1 = '%(asctime)s - %(name)s: %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'

    FORMATS = {
        logging.DEBUG: LIGHTGREY + format1 + RESET,
        logging.INFO: GREEN + format0 + RESET,
        logging.WARNING: YELLOW + format1 + RESET,
        logging.ERROR: RED + format1 + RESET,
        logging.CRITICAL: BOLD + RED + format1 + RESET,
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
