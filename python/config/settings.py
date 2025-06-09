#!/usr/bin/env python3
from dataclasses import fields
from pathlib import Path
from typing import (
    Generic,
    Self,
    TypeVar,
    override,
)

from pydantic.dataclasses import dataclass
from util.exceptions import ConfigurationError
from util.logger import get_logger

# NOTE: Use pydantic for verification when loading from files.

logger = get_logger(__name__)

T = TypeVar('T', bound='Settings')
U = TypeVar('U', bound='DefaultSettings')


@dataclass
class Settings:
    """
    Base class for configuration settings.
    Supports conversion to/from dictionaries with special handling for Path
    objects. Only supports primitive types, Path objects, and lists of
    primitives/Paths.
    The TOML representation is automatically flattened when loading.
    """

    def export_dict(self) -> dict:
        """
        Convert settings to a dictionary.
        `Path` and `list[Path]` are converted to `str` and `list[str]`.

        :return: Dictionary representation of settings.
        """
        return {
            fld.name: (
                str(v)
                if isinstance(v, Path)
                else list(map(str, v))
                if (v and isinstance(v, list) and isinstance(v[0], Path))
                else v
            )
            for fld in fields(self)
            if (v := getattr(self, fld.name)) is not None
        }

    @classmethod
    def load_dict(cls, conf: dict) -> Self:
        """
        Create settings instance from a dictionary.
        `Path` and `list[Path]` are created from `str` and `list[str]` when
        nessecary.

        :param conf: Dictionary of configuration values.
        :return: Settings instance populated from dictionary.
        :raises ConfigurationError: If dictionary structure does not match
            settings field types or constraints.
        """
        # Build the object.
        settings = cls(
            **{
                fld.name: (
                    Path(v)
                    if fld.type is Path
                    else list(map(Path, v))
                    if fld.type == list[Path]
                    else v
                )
                for fld in fields(cls)
                if (v := conf[fld.name]) is not None
            }
        )

        logger.debug(f'Loaded {cls.__name__}.')
        debug_contents_str = f'{cls.__name__} contents:\n'
        for k, v in settings.export_dict().items():
            debug_contents_str += f'{k} = {v}\n'
        logger.debug(debug_contents_str)
        return settings


@dataclass
class DefaultSettings(Settings, Generic[T]):
    """Provides default values for settings classes."""

    @classmethod
    def __post_init__(cls):
        a = cls
        b = cls._get_base()
        a_fields = set((fld.name, fld.type) for fld in fields(cls))
        b_fields = set((fld.name, fld.type) for fld in fields(cls._get_base()))
        ab = a_fields - b_fields
        ba = b_fields - a_fields
        if ab:
            raise ConfigurationError(
                f'Extra fields in default settings: {ab}, in {a}/{b}'
            )
        if ba:
            raise ConfigurationError(
                f'All fields must be explicitly declared: {ba}, in {a}/{b}'
            )

    @classmethod
    def _get_base(cls) -> type[T]:
        """
        Extract concrete settings type from generic parameter.

        :return: The concrete settings class type.
        :rtype: type[T]
        :raises ConfigurationError: If constraints not followed.
        """
        for base in getattr(cls, '__orig_bases__', ()):
            if hasattr(base, '__args__') and base.__args__:
                arg = base.__args__[0]
                if not isinstance(arg, TypeVar):
                    return arg

        raise ConfigurationError('Do not use bare `DefaultSettings`.')

    @override
    @classmethod
    def load_dict(cls, conf: dict) -> T:
        """
        Create settings instance with default values as fallback.

        :param conf: Configuration dictionary to overlay on defaults.
        :return: Combined settings instance.
        :rtype: T
        """
        return cls._get_base().load_dict(cls().export_dict() | conf)
