#!/usr/bin/env python3
############################################################################
#                                                                          #
#  Copyright (C) 2025                                                      #
#                                                                          #
#  This program is free software: you can redistribute it and/or modify    #
#  it under the terms of the GNU General Public License as published by    #
#  the Free Software Foundation, either version 3 of the License, or       #
#  (at your option) any later version.                                     #
#                                                                          #
#  This program is distributed in the hope that it will be useful,         #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#  GNU General Public License for more details.                            #
#                                                                          #
#  You should have received a copy of the GNU General Public License       #
#  along with this program. If not, see <http://www.gnu.org/licenses/>.    #
#                                                                          #
############################################################################
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    ForwardRef,
    Generic,
    Self,
    TypeVar,
)

import tomli_w
import tomllib
from configuration.settings import (
    T,
    U,
)
from configuration.structure import (
    DefaultGlobalSettings,
    GlobalSettings,
    V,
    W,
    X,
    Y,
)
from pydantic import ValidationError
from utils.exceptions import ConfigurationError
from utils.logger import get_logger

logger = get_logger(__name__)

_sfx = '.toml'


P = TypeVar('P', bound='InstallationConfig')
Q = TypeVar('Q', bound='GameConfig')


@dataclass
class Config(Generic[T, U]):
    """
    Configuration container for settings.

    :ivar path: Path to the configuration file.
    :ivar settings: Loaded settings instance.
    :vartype settings: T
    """

    path: Path
    settings: T

    # Adopted children.
    @classmethod
    def _get_ubase(cls) -> type[T]:
        """
        Get the settings type from generic parameters.

        :return: Settings type.
        :rtype: type[T]
        :raises TypeError: See `_get_uchild` and `DefaultSettings._get_base`.
        :raises ConfigurationError: See `_get_uchild` and
        `DefaultSettings._get_base`.
        """
        return cls._get_u()._get_base()

    @classmethod
    def _get_u(cls) -> type[U]:
        """
        Get the default settings type from generic parameters.

        :return: Default settings type.
        :rtype: type[U]
        :raises TypeError: If generic parameters are invalid.
        :raises ConfigurationError: If generic not subclassed.
        """
        for base in getattr(cls, '__orig_bases__', ()):
            if hasattr(base, '__args__') and len(base.__args__) > 1:
                arg = base.__args__[1]
                if not isinstance(arg, TypeVar):
                    return arg
            else:
                raise TypeError(f'Bad base (len <= 1): {base}')

        raise ConfigurationError('Do not use bare `Config`.')

    @classmethod
    def _get_t_u_tup(cls) -> tuple[type[T], type[U]]:
        """
        Get both settings and default settings types.

        :return: Settings and default settings types as a tuple.
        :rtype: tuple[type[T], type[U]]
        :raises TypeError: See `_get_uchild` and `DefaultSettings._get_base`.
        :raises ConfigurationError: See `_get_uchild` and
        `DefaultSettings._get_base`.
        """
        uchild = cls._get_u()
        return uchild._get_base(), uchild

    @classmethod
    def flatten_dict(cls, conf: dict[str, Any]) -> dict:
        """
        Flatten a dictionary with at most one level of nesting.
        Inverse operation of `restore_dict` for the restrictions:
            - Top-level keys never have underscores.
            - Top-level keys never contain dicts.
            - Table keys never have underscores.

        :param conf: Configuration dictionary with at most one nested level
        :param prefix: Current key prefix (used for nested contexts)
        :return: Flattened single-level dictionary
        """
        flat_dict = {}
        for k, v in conf.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    flat_dict[f'{k}_{sk}'] = sv
            else:
                flat_dict[k] = v
        return flat_dict

    @classmethod
    def restore_dict(cls, flat_conf: dict) -> dict:
        """
        Restore a nested dictionary from a flattened representation.
        Inverse operation of `flatten_dict` for the restrictions:
            - Top-level keys never have underscores.
            - Top-level keys never contain dicts.
            - Table keys never have underscores.

        :param flat_conf: Flattened dictionary with keys in `section_key` format
        :return: Nested dictionary with top-level sections

        Example:
            Input: {'foo_bar': 114, 'foo_baz': 514, 'top': 42}
            Output: {'foo': {'bar': 114, 'baz': 514}, 'top': 42}
        """
        nested_dict = {}

        for key, value in flat_conf.items():
            # Split key at first underscore if present.
            if '_' in key:
                section, subkey = key.split('_', 1)

                # Create section if not exists.
                if section not in nested_dict:
                    nested_dict[section] = {}

                # Add value to section dictionary.
                nested_dict[section][subkey] = value
            else:
                # Top-level keys without underscores remain at root.
                nested_dict[key] = value

        return nested_dict

    @staticmethod
    def tomllib_load_or_create(path: Path) -> dict:
        """
        Load TOML configuration file to a dictionary.
        This will attempt to create the file and its parent directories.

        :param path: Path to TOML file
        :return: Parsed configuration dictionary
        :raises ConfigurationError: If file parsing fails
        """
        conf: dict[str, Any]
        try:
            path.parents[0].mkdir(parents=True, exist_ok=True)  # Mkdir parents.
            path.touch(exist_ok=True)  # Touch file.
            with open(path, 'rb') as f:
                conf = tomllib.load(f)
        except Exception as e:
            raise ConfigurationError(f'Failed to load file {path}: {e}') from e

        return conf

    @classmethod
    def load_toml(cls, config_path: Path, DSettings: type[U]) -> Self:
        """
        Create configuration from TOML file with default fallback.

        :param config_path: Path to configuration file
        :param DSettings: Default settings provider
        :return: Config instance with loaded settings
        :rtype: Self
        """
        conf = cls.flatten_dict(cls.tomllib_load_or_create(config_path))
        try:
            settings: T = DSettings.load_dict(conf)
        except ValidationError as e:
            raise ConfigurationError(f'Badly typed value: {e}') from e

        # `ForwardRef`s have `__forward_arg__` instead of `__name__`.
        u_name = (
            u.__forward_arg__
            if isinstance(u := cls._get_u(), ForwardRef)
            else u.__name__
        )
        logger.debug(f'Loaded {cls.__name__} from {u_name}.')
        return cls(config_path, settings)

    def write_toml(self, path: Path | None = None) -> None:
        """
        Write the current settings to a TOML file.
        This will attempt to create the file and its parent directories.
        See `restore_dict`.

        :param file: The path to the TOML file where the settings will be written.
        """
        if path is None:
            path = self.path
        path.parents[0].mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)  # Touch file.
        logger.debug(f'Write {self._get_ubase().__name__} to {path}.')
        with open(path, 'wb') as f:
            tomli_w.dump(self.restore_dict(self.settings.export_dict()), f)


@dataclass
class GlobalConfig(Config[GlobalSettings, DefaultGlobalSettings]):
    """Global configuration container."""

    settings: GlobalSettings


@dataclass
class InstallationConfig(Config[V, W]):  # V~ InstallationSettings, Y~ Default"
    """Installation-specific configuration container."""

    settings: V  # A subclass of InstallationSettings.


@dataclass
class GameConfig(Config[X, Y]):  # X~ GameSettings, Y~ Default"
    """Game-specific configuration container."""

    settings: X  # A subclass of GameSettings.


@dataclass
class ConfigHolder(Generic[P, Q, V, X]):
    """
    Aggregate container for all configuration levels.

    :ivar glob: Global configuration
    :ivar inst: Installation configuration
    :ivar game: Game-specific configuration
    """

    glob: GlobalConfig
    inst: P
    game: Q

    def __post_init__(self) -> None:
        self.globs: GlobalSettings = self.glob.settings
        self.insts: V = self.inst.settings
        self.games: X = self.game.settings

    def export_conf3(self) -> None:
        """Create configuration files (and parent directories)."""
        logger.debug('Attempt to export 3 configuration files.')
        self.glob.write_toml()
        self.inst.write_toml()
        self.game.write_toml()
        logger.debug('Successfully exported 3 configuration files.')


def load_conf3(
    config_path: Path,
    project_name: str,
    IConf: type[InstallationConfig],
    GConf: type[GameConfig],
    _DGS: type[DefaultGlobalSettings] = DefaultGlobalSettings,
    export_after_load: bool = True,
) -> ConfigHolder:
    # TODO: Allow loading from user-defined toml s.t.
    # defaults <- config (overwritten each time) <- user config (persistant)
    # XXX: Above is probably bad.
    """
    Load hierarchical configuration.
    This will (re)create the toml files, overwriting all comments.

    :param config_path: Path to global configuration file
    :param project_name: Project identifier to load
    :param IC: `InstallationConfig` subclass type
    :param GC: `GameConfig` subclass type
    :param _DGlobS: `DefaultGlobalSettings` subclass type. Example only.
    :return: ConfigHolder with all configuration levels
    :raises ConfigurationError: If requested game isn't registered
    """
    logger.debug('Attempt to load 3 configuration files.')
    ISettings, DISettings = IConf._get_t_u_tup()
    GSettings, DGSettings = GConf._get_t_u_tup()

    # Read global config.
    global_config = GlobalConfig.load_toml(config_path, _DGS)

    # Check if game exists.
    if project_name not in (project_names := global_config.settings.projects):
        raise ConfigurationError(
            f'{project_name} not installed or not registered in {project_names}'
        )

    # Read installation config.
    installation_config_path = global_config.settings.data_dir.joinpath(
        project_name
    ).with_suffix(_sfx)
    installation_config = IConf.load_toml(installation_config_path, DISettings)

    # Read game config.
    game_config_path = installation_config.settings.data_game_config
    game_config = GConf.load_toml(game_config_path, DGSettings)

    logger.debug('Successfully loaded 3 configuration files.')
    confh = ConfigHolder[IConf, GConf, ISettings, GSettings](
        global_config, installation_config, game_config
    )

    # Export after load.
    if export_after_load:
        confh.export_conf3()

    return confh
