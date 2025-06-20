#!/usr/bin/env python3
from pprint import pp

from llm.utils.context_manager import (
    Context,
    RegisterHook,
)
from llm.utils.stream import (
    ContentStr,
    CustomHookArgs,
    CustomStreamHook,
    ReasoningContentStr,
)
from world.character_roles import RoleProfiles
from world.character.character_class import Character
from utils.ansi import DARKGREY, END, GREEN
from utils.logger import get_logger

from examples.config import confh

logger = get_logger(__name__)


rp = RoleProfiles(confh)


@CustomStreamHook
def print_each_word_hook(args: CustomHookArgs) -> None:
    head = args.head_as_str
    if isinstance(head, ReasoningContentStr):
        print(DARKGREY + args.head_as_str + END, end='', flush=True)
    elif isinstance(head, ContentStr):
        print(GREEN + args.head_as_str + END, end='', flush=True)


@CustomStreamHook
def cat_hook(ha: CustomHookArgs) -> None:
    if '喵' in ha.head_as_str.lower():
        logger.critical('(ฅ>ω<ฅ)')


@RegisterHook
def example_register_hook(x: Character, y: Context):
    pp(x)
    pp(y)
