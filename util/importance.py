#!/usr/bin/env python3
from enum import StrEnum
from functools import total_ordering
from typing import Any, TypeVar, final

# NOTE: Metaclass hinting is ignored for pyright only.


@final
class imauto:  # Equivalent of `enum.auto`.
    """Class for generating enum members with case-preserving values."""

    def __init__(self):
        self.value = None

    def __set_name__(self, _, name: str):
        self.value = name

    # Inform type checkers of the accessed class.
    def __get__(self, _: Any, owner: type['Importance']) -> 'ImportanceEnum':
        return owner._enum_class[self.value]  # type: ignore

    def __repr__(self):
        return f'<imauto: {self.value}>'


class classproperty(property):
    """Decorator for class properties."""

    def __get__(self, _, owner=None) -> Any:
        """Retrieve the property value for the class.

        :param _: Unused.
        :param owner: The owning class.
        :return: The property value.
        """
        return classmethod(self.fget).__get__(None, owner)()  # pyright: ignore[reportArgumentType]


@total_ordering
class ImportanceFloat:
    """Wrapper for integer values used in `Importance` comparisons."""

    __slots__ = ['value']

    def __init__(self, value: float) -> None:
        """
        Initialize with a (real) numeric value.

        :param value: The numeric value to wrap.
        :type value: float
        """
        self.value = value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ImportanceFloat):
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, ImportanceFloat):
            return self.value < other.value
        return NotImplemented

    def __repr__(self) -> str:
        return f'ILevel({self.value})'


class ImportanceEnum(StrEnum):
    """
    Enum where members are also (and must be) strings, and are totally ordered
    top-to-bottom.
    """

    @classmethod
    def order(cls) -> tuple['ImportanceEnum', ...]:
        return tuple(e for e in cls)

    @staticmethod
    def order_eq(fst, snd: Any) -> bool:
        if not isinstance(snd, ImportanceEnum):
            return NotImplemented
        return fst is snd

    # NOTE: Do not use `@total_ordering` to avoid violating LSP
    @staticmethod
    def order_lt(fst, snd: Any) -> bool:
        if not isinstance(snd, ImportanceEnum):
            return NotImplemented
        order = fst.order()
        return order.index(fst) < order.index(snd)

    @classmethod
    def order_le(cls, fst, snd: Any) -> bool:
        return cls.order_lt(fst, snd) or cls.order_eq(fst, snd)

    @classmethod
    def order_gt(cls, fst, snd: Any) -> bool:
        return not cls.order_lt(fst, snd)

    @classmethod
    def order_ge(cls, fst, snd: Any) -> bool:
        return not cls.order_lt(fst, snd) or cls.order_eq(fst, snd)


T = TypeVar('T', bound=ImportanceEnum)


class ImportanceType(type):
    """
    Metaclass for Importance classes that automatically creates enum-based
    properties.

    :param isplit: The index of the enum member to use as the split point for
        comparisons.
    :type isplit: int
    """

    _enum_class: ImportanceEnum  # Actual: ClassVar[type[ImportanceEnum]]
    _split: ImportanceEnum

    @staticmethod
    def ILevel(i: Any) -> Any:
        _ = i  # Unused.

    def __new__(cls, name, bases, namespace, isplit: int = 0, **attrs):
        _ = attrs  # Unused.
        new_class = super().__new__(cls, name, bases, namespace)

        if name == 'Importance':
            return new_class

        # NOTE: `imauto does not automatically lowercase names for
        # `ImportanceEnum` like `enum.auto` does for `StrEnum`.
        enum_members = {
            name: value if not _is_str else name
            for name, value in namespace.items()
            if not name.startswith('_')
            if (_is_str := isinstance(value, str) or isinstance(value, imauto))
        }

        enum_class = ImportanceEnum(name + 'Enum', enum_members)
        new_class._enum_class = enum_class
        try:
            new_class._split = enum_class.order()[isplit]
        except IndexError as e:
            raise IndexError(
                f'Key `isplit` ({isplit}) is out of range when indexing the'
                f' defined `ImportanceEnum` (length {len(enum_class)}).'
            ) from e

        # Add properties for each enum value.
        for level in enum_class:  # pyright: ignore [reportAssignmentType]
            level: ImportanceEnum
            setattr(
                new_class,
                level.name,
                classproperty(lambda _, lvl=level: new_class(lvl)),
            )

        # Add the IMPORTANCE method.
        new_class.ILevel = staticmethod(lambda i: new_class(ImportanceFloat(i)))

        return new_class

    def __repr__(self) -> str:
        return f"<imenum '{self.__name__}'>"


@total_ordering
class Importance(metaclass=ImportanceType):
    """
    Base Importance class that can be subclassed with custom enum values.
    """

    def __init__(self, i: ImportanceEnum | ImportanceFloat) -> None:
        self.i = i

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Importance):
            return NotImplemented

        if isinstance(self.i, ImportanceFloat) and isinstance(
            other.i, ImportanceFloat
        ):
            return self.i == other.i
        elif isinstance(self.i, ImportanceEnum) and isinstance(
            other.i, ImportanceEnum
        ):
            return ImportanceEnum.order_eq(self, other)
        return False

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Importance):
            return NotImplemented
        si, oi = self.i, other.i
        split = self._split

        if isinstance(si, ImportanceFloat) and isinstance(oi, ImportanceFloat):
            return si < oi
        elif isinstance(si, ImportanceEnum) and isinstance(oi, ImportanceEnum):
            return ImportanceEnum.order_lt(si, oi)
        elif isinstance(si, ImportanceFloat):
            return ImportanceEnum.order_gt(oi, split)
        else:
            return ImportanceEnum.order_le(si, split)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}.{self.i}'
