#!/usr/bin/env python3
from dataclasses import dataclass
from enum import EnumType, StrEnum
from typing import Any, Container

# NOTE: Metaclass hinting is ignored for pyright only.


class ICBEnumType(EnumType):
    """Metaclass for IdClassBase."""

    # Avoid `__or__` polymorphism.
    def __add__(self, o: 'ICBEnumType') -> 'ICBEnumType':
        """
        Combine two `ICBEnumType` metaclasses into a new one.

        :param o: Another `ICBEnumType` metaclass.
        :return: A new metaclass combining the entries of both.
        :rtype: ICBEnumType
        """
        name = f'{self.__name__}_{o.__name__}'
        entries = {}
        for e in (self, o):
            for m in e:
                entries[m.name] = m.value  # pyright: ignore [reportAttributeAccessIssue]

        return IdClassBase(name, entries)  # pyright: ignore [reportReturnType]


class IdClassBase(StrEnum, metaclass=ICBEnumType):
    def __contains__(self, key: Any) -> bool:
        """
        Check if the key is an `AraId` of this class.

        :param key: The object to check.
        :return: `True` if `key` is an `AraId` of this class, else `False`.
        :rtype: bool
        :raises TypeError: If the key is not an instance of `AraId`.
        """
        if not isinstance(key, AraId):
            raise TypeError(f'Key ({key}) is not an instance of `AraId`.')
        return key.id_class == self


@dataclass(frozen=True)
class AraId(Container):
    """
    Represents an identifier with a class and a number.

    :param id_class: The class/category of the identifier.
    :type id_class: IdClassBase
    :param id_num: The numerical value of the identifier.
    :type id_num: int
    """

    id_class: IdClassBase
    id_num: int

    def __contains__(self, key: Any) -> bool:
        """
        Check if the key matches the key number.

        :param key: The key to check against the identifier.
        :type key: int
        :return: True if the key matches, False otherwise.
        :rtype: bool
        """
        if not isinstance(key, int):
            raise TypeError(f'Key ({key}) is not an integer.')

        return key == self.id_num


class AraIdSupply:
    """
    A unique identifier supply for given classes.

    :param id_classes: The set of identifier classes to manage.
    :type id_classes: ICBEnumType
    :ivar counter: A counter for each identifier class.
    :type counter: dict[IdClassBase, int]
    """

    def __init__(self, id_classes: ICBEnumType) -> None:
        """
        Initialize the identifier supply with the given classes.

        :param id_classes: The enum of identifier classes to manage.
        :type id_classes: ICBEnumType
        """
        self.id_classes = id_classes
        self.counter = {ie: 0 for ie in self.id_classes}

    def next_id(self, id_class: IdClassBase) -> AraId:
        """
        Generate the next identifier for the specified class.

        :param id_class: The class for which to generate the identifier.
        :type id_class: IdClassBase
        :return: A new identifier with the next available number for the class.
        :rtype: AraId
        :raises ValueError: If the class is not managed by this supply.
        """
        if id_class in self.id_classes:
            self.counter[id_class] += 1  # pyright: ignore [reportArgumentType]
            return AraId(id_class, self.counter[id_class])  # pyright: ignore [reportArgumentType]
        else:
            raise ValueError(
                f'Received ID class {id_class} not in {self.id_classes}'
            )

    def next_ids(self, id_class: IdClassBase, n: int) -> tuple[AraId, ...]:
        """
        Generate the next `n` identifiers for the specified class.

        :param id_class: The class for which to generate the identifier.
        :type id_class: IdClassBase
        :param n: The amount of identifiers to generate.
        :type n: int
        :return: `n` new identifiers for the class.
        :rtype: tuple[AraId, ...]
        :raises ValueError: See `AraIdSupply.next_id`
        """
        return tuple(self.next_id(id_class) for _ in range(n))

    @staticmethod
    def get_base() -> tuple[ICBEnumType, type]:
        """
        Get the base enum and metaclass for identifier classes.

        :return: The base enum and metaclass for identifier classes.
        :rtype: tuple[ICBEnumType, Type]
        """
        return IdClassBase, ICBEnumType
