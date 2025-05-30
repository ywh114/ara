#!/usr/bin/env python3
from dataclasses import dataclass, fields
from pathlib import Path
from types import GenericAlias, UnionType
from typing import (
    Generic,
    Self,
    TypeVar,
    get_args,
    get_origin,
    override,
)

from util.exceptions import ConfigurationError
from util.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T', bound='Settings')
U = TypeVar('U', bound='DefaultSettings')


_settings_item_types_allowed_tuple = (
    str,
    int,
    float,
    bool,
    Path,
    list[str],
    list[int],
    list[float],
    list[bool],
    list[Path],
)


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
        flds = fields(cls)
        try:
            # Keys must match.
            conf_keyset = set(conf.keys())
            cls_keyset = set(fld.name for fld in flds)
            if conf_keyset != cls_keyset:
                raise KeyError(
                    f'Keys do not match: \n\tDictionary keys: {conf_keyset}'
                    f'\n\t{cls} keys: {cls_keyset}'
                )

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
            # Enforce dataclass types.
            # We only expect list[str], list[int], list[float], or list[bool]
            # as `GenericAlias`es.
            # TODO: Rewrite.
            allowed = _settings_item_types_allowed_tuple
            for fld in flds:
                v = getattr(settings, fld.name)
                _T = fld.type
                if not isinstance(_T, (type, GenericAlias)):
                    raise TypeError(f'Bad annotation: {_T}')
                elif not any(_T == t for t in allowed):
                    raise TypeError(f'Field type ({_T}) not in {allowed}.')
                try:
                    if isinstance(_T, GenericAlias):
                        if isinstance(v, origin := get_origin(_T)):
                            args = get_args(_T)
                            if len(v) == 0:
                                continue  # Allow empty lists.
                            elif len(args) != 1:
                                raise TypeError(
                                    'The origin must have exactly one arg, not '
                                    f'[{args}]'
                                )
                            elif isinstance(a0 := args[0], UnionType):
                                raise TypeError(
                                    f'{origin}[arg]: arg should be a type, not '
                                    f'`UnionType` {args}'
                                )
                            elif not all(
                                isinstance(x, a0) for x in v
                            ):  # Enforce `origin[arg[0]]`.
                                raise TypeError(
                                    'All elements must be of the same type: '
                                    f'{a0}, not '
                                    f'{tuple(type(x) for x in v)}'
                                )
                            else:
                                # Check passed.
                                continue
                    elif not isinstance(v, _T):
                        raise TypeError(
                            f'{v} ({type(v)}) is not an instanceof {_T}'
                        )
                except TypeError as e:
                    raise ConfigurationError(
                        f'Types do not match:'
                        f'\n\tGot: {v} ({type(v)})'
                        f'\n\tExpected: ({_T})'
                        f'\n\tDetails: {e}'
                    ) from e
        except (TypeError, KeyError) as e:
            raise ConfigurationError(
                f'Dictionary entries must match {cls}: {e}'
            ) from e

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
