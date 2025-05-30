#!/usr/bin/env python3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Container, Generic, TypeVar
from pathlib import Path

T = TypeVar('T')
U = TypeVar('U')


@dataclass
class KSearchSpec:
    pass


@dataclass
class KReturnSpec:
    pass


class KData(Container[KReturnSpec]):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def __contains__(self, item: Any) -> bool:
        """Check if the item is contained in the Data instance.

        Args:
            item: The item to check for presence in the container.

        Returns:
            bool: True if the item is found, False otherwise.
        """
        if not isinstance(item, KSearchSpec):
            raise TypeError(f'Item {item} is not an instance of {KSearchSpec}')

        raise NotImplementedError('WIP')  # TODO: implement

    def __getitem__(self, key: Any) -> KReturnSpec:
        """Retrieve an item from the container.

        Args:
            key: The key to retrieve the item.

        Returns:
            KReturnSpec: The retrieved item.
        """
        raise NotImplementedError('WIP')  # TODO: implement


@dataclass
class Knowledge(ABC, Generic[T, U]):
    domain_name: str
    data: Container[U]

    @classmethod
    @abstractmethod
    def retrieve(cls, spec: T) -> U: ...


@dataclass
class GatedKnowledge(Knowledge[KSearchSpec, KReturnSpec]):
    data: Container[KReturnSpec]
