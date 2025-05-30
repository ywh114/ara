#!/usr/bin/env python3
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import GenericAlias
from typing import (
    Any,
    Generic,
    Self,
    TypeVar,
    get_args,
    get_origin,
    override,
)

import tomli_w
import tomllib
import xdg_base_dirs as xdg

T = TypeVar('T', bound='Settings')
U = TypeVar('U', bound='DefaultSettings')


V = TypeVar('V', bound='InstallationSettings')
W = TypeVar('W', bound='DefaultInstallationSettings')


X = TypeVar('X', bound='GameSettings')
Y = TypeVar('Y', bound='DefaultGameSettings')


@dataclass
class GlobalInfo:
    """
    Global application constants and paths.

    :var prefix: Application name prefix used in paths.
    :var config_dir: Base directory for configuration files.
    :var config_path: Main configuration file path.
    """

    prefix = 'ara'
    config_dir: Path = xdg.xdg_config_home().joinpath(prefix)
    config_path: Path = config_dir.joinpath(f'{prefix}.toml')


SettingsItemTypes = (
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

    def to_dict(self) -> dict:
        """
        Convert settings to a dictionary.

        :return: Dictionary representation of settings with Paths converted to
            strings.
        :rtype: dict
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
    def from_dict(cls: type[Self], conf: dict) -> Self:
        """
        Create settings instance from a dictionary.

        :param conf: Dictionary of configuration values.
        :type conf: dict
        :return: Settings instance populated from dictionary.
        :rtype: Self
        :raises TypeError: If dictionary structure doesn't match settings
            fields.
        """
        try:
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
            # Check types.
            # We only expect list[str], list[int], list[float], or list[bool].
            for fld in fields(cls):
                v = getattr(settings, fld.name)
                T = fld.type
                if not isinstance(T, (type, GenericAlias)):
                    raise TypeError(f'{T} is not a type.')
                try:
                    if isinstance(T, GenericAlias):
                        if isinstance(v, get_origin(T)):
                            if len(v) == 0:
                                # Allow empty lists.
                                continue
                            elif len(args := get_args(T)) == 1 and all(
                                isinstance(e, args[0]) for e in v
                            ):
                                continue
                            else:
                                raise TypeError
                    elif not isinstance(v, T):
                        raise TypeError
                except TypeError as e:
                    raise TypeError(
                        f'Types do not match:'
                        f'\n\tGot: {v} ({type(v)})'
                        f'\n\tExpected: ({T})'
                    ) from e
        except TypeError as e:
            raise TypeError(f'Dictionary entries must match {cls}: {e}') from e

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
        :raises RuntimeError: If used without generic type parameter.
        """
        for base in getattr(cls, '__orig_bases__', ()):
            if hasattr(base, '__args__') and base.__args__:
                arg = base.__args__[0]
                if not isinstance(arg, TypeVar):
                    return arg

        raise RuntimeError('Do not use bare `DefaultSettings`.')

    @override
    @classmethod
    def from_dict(cls, conf: dict) -> T:
        """
        Create settings instance with default values as fallback.

        :param conf: Configuration dictionary to overlay on defaults.
        :type conf: dict
        :return: Combined settings instance.
        :rtype: T
        """
        return cls._get_base().from_dict(cls().to_dict() | conf)


@dataclass
class GlobalSettings(Settings):
    """
    Global application settings.

    :ivar data_dir: Base directory for application data.
    :vartype data_dir: Path
    :ivar cache_dir: Base directory for cached files.
    :vartype cache_dir: Path
    :ivar projects: List of installed project identifiers.
    :vartype projects: list[str]
    """

    data_dir: Path
    cache_dir: Path
    projects: list[str]


@dataclass
class DefaultGlobalSettings(DefaultSettings[GlobalSettings]):
    """Default values for global settings using XDG base directories."""

    data_dir: Path = xdg.xdg_data_home().joinpath(GlobalInfo.prefix)
    cache_dir: Path = xdg.xdg_cache_home().joinpath(GlobalInfo.prefix)
    projects: list[str] = field(default_factory=list)


GlobS, DGlobS = GlobalSettings, DefaultGlobalSettings


# NOTE: Subclass from installation and game settings. Avoid modifying any global
# settings unless for testing/an example; use `DefaultGlobalSettings`.
# XXX: All fields must be manually filled in when subclassing. There is no
# automatic concatanation of paths.


@dataclass
class InstallationSettings(Settings):
    """
    Base class for per-installation settings for game projects.
    Installation settings must at least contain the below:

    :ivar project_name: Identifier for the project.
    :vartype project_name: str
    :ivar project_data_dir: Project-specific data directory.
    :vartype project_data_dir: Path
    :ivar project_cache_dir: Project-specific cache directory.
    :vartype project_cache_dir: Path
    :ivar data_game_dir: Directory for game data files.
    :vartype data_game_dir: Path
    :ivar data_game_config: Path to game-specific configuration.
    :vartype data_game_config: Path
    :ivar cache_assets_dir: Directory for cached assets.
    :vartype cache_assets_dir: Path
    :ivar embedding_db_dir: Directory for vector databases.
    :vartype embedding_db_dir: Path
    :ivar embedding_model_dir: Directory for embedding models.
    :vartype embedding_model_dir: Path
    """

    # [project]
    project_name: str
    project_data_dir: Path
    project_cache_dir: Path

    # [data]
    data_game_dir: Path
    data_game_config: Path

    # [cache]
    cache_assets_dir: Path

    # [embedding]
    embedding_db_dir: Path
    embedding_models_dir: Path


@dataclass
class DefaultInstallationSettings(DefaultSettings[V]):
    """Default values for installation settings."""

    # [project]
    project_name: str = 'default_project'
    project_data_dir: Path = DGlobS.data_dir.joinpath(project_name)
    project_cache_dir: Path = DGlobS.cache_dir.joinpath(project_name)

    # [data]
    data_game_dir: Path = project_data_dir.joinpath('game')
    data_game_config: Path = project_data_dir.joinpath(f'{project_name}.toml')

    # [cache]
    cache_assets_dir: Path = project_cache_dir.joinpath('assets')

    # [embedding]
    embedding_db_dir: Path = cache_assets_dir.joinpath('db')
    embedding_models_dir: Path = DGlobS.cache_dir.joinpath('embedding_models')


@dataclass
class GameSettings(Settings):
    """Base class for game-specific settings."""

    pass


@dataclass
class DefaultGameSettings(DefaultSettings[X]):
    """Base class for default game-specific settings."""

    pass


@dataclass
class Config(Generic[T, U]):
    """
    Configuration container for settings.

    :ivar path: Path to the configuration file.
    :vartype path: Path
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
        :raises RuntimeError: See `_get_uchild` and `DefaultSettings._get_base`.
        """
        return cls._get_u()._get_base()

    @classmethod
    def _get_u(cls) -> type[U]:
        """
        Get the default settings type from generic parameters.

        :return: Default settings type.
        :rtype: type[U]
        :raises RuntimeError: If generic parameters are invalid.
        """
        for base in getattr(cls, '__orig_bases__', ()):
            if hasattr(base, '__args__') and len(base.__args__) > 1:
                arg = base.__args__[1]
                if not isinstance(arg, TypeVar):
                    return arg
            else:
                raise RuntimeError(f'Bad base (len <= 1): {base}')

        raise RuntimeError('Do not use bare `Config`.')

    @classmethod
    def _get_t_u_tup(cls) -> tuple[type[T], type[U]]:
        """
        Get both settings and default settings types.

        :return: Settings and default settings types as a tuple.
        :rtype: tuple[type[T], type[U]]
        :raises RuntimeError: See `_get_uchild` and `DefaultSettings._get_base`.
        """
        uchild = cls._get_u()
        return uchild._get_base(), uchild

    @classmethod
    def flatten_dict(cls, conf: dict[str, Any], prefix: str = '') -> dict:
        """
        Recursively flatten nested dictionary structure.
        Note that we expect only depth 1 nesting for our usecase.

        :param conf: Nested configuration dictionary
        :type conf: dict[str, Any]
        :param prefix: Current key prefix (used recursively)
        :type prefix: str
        :return: Flattened single-level dictionary
        :rtype: dict
        """
        flat_dict = {}

        for key, value in conf.items():
            # Append the keyname to the prefix.
            full_key = f'{prefix}{key}'
            if isinstance(value, dict):
                # Note we add a trailing underscore.
                flat_dict.update(cls.flatten_dict(value, f'{full_key}_'))
            else:
                flat_dict[full_key] = value

        return flat_dict

    @classmethod
    def restore_dict(cls, flat_conf: dict) -> dict:
        """
        Restore a nested dictionary from a flattened representation.
        Inverse operation of `flatten_dict` for the restrictions:
            - The dictionary is nested to at most depth 1.
            - Top-level keys never have underscores.
            - Table keys never have underscores.

        :param flat_conf: Flattened dictionary with keys in `section_key` format
        :type flat_conf: dict
        :return: Nested dictionary with top-level sections
        :rtype: dict

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
    def load_toml(path: Path) -> dict:
        """
        Load TOML configuration file to a dictionary.
        This will attempt to create the file and its parent directories.

        :param path: Path to TOML file
        :type path: Path
        :return: Parsed configuration dictionary
        :rtype: dict
        :raises RuntimeError: If file parsing fails
        """
        conf: dict[str, Any]
        try:
            path.parents[0].mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)  # Touch file.
            with open(path, 'rb') as f:
                conf = tomllib.load(f)
        except Exception as e:
            raise RuntimeError(f'Failed to parse file {path}: {e}') from e

        return conf

    @classmethod
    def from_toml(cls, config_path: Path, DSettings: type[U]) -> Self:
        """
        Create configuration from TOML file with default fallback.

        :param config_path: Path to configuration file
        :type config_path: Path
        :param DSettings: Default settings provider
        :type DSettings: U
        :return: Config instance with loaded settings
        :rtype: Self
        """
        conf = cls.flatten_dict(cls.load_toml(config_path))
        settings: T = DSettings.from_dict(conf)

        return cls(config_path, settings)

    def write_toml(self, path: Path | None = None) -> None:
        """
        Write the current settings to a TOML file.
        This will attempt to create the file and its parent directories.
        See `restore_dict`.

        :param file: The path to the TOML file where the settings will be written.
        :type file: Path
        """
        if path is None:
            path = self.path
        path.parents[0].mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)  # Touch file.
        with open(path, 'wb') as f:
            tomli_w.dump(self.restore_dict(self.settings.to_dict()), f)


@dataclass
class GlobalConfig(Config[GlobS, DGlobS]):
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
class ConfigHolder:
    """
    Aggregate container for all configuration levels.

    :ivar glob: Global configuration
    :vartype glob: GlobalConfig
    :ivar inst: Installation configuration
    :vartype inst: InstallationConfig
    :ivar game: Game-specific configuration
    :vartype game: GameConfig
    """

    glob: GlobalConfig
    inst: InstallationConfig
    game: GameConfig


def loadconf(
    config_path: Path,
    project_name: str,
    IC: type[InstallationConfig],
    GC: type[GameConfig],
    _DGlobS: type[DefaultGlobalSettings] = DefaultGlobalSettings,
) -> ConfigHolder:
    # TODO: Allow loading from user-defined toml s.t.
    # defaults <- config (overwritten each time) <- user config (persistant)
    """
    Load hierarchical configuration.
    This will (re)create the toml files, overwriting all comments.

    :param config_path: Path to global configuration file
    :type config_path: Path
    :param project_name: Project identifier to load
    :type project_name: str
    :param IC: `InstallationConfig` subclass type
    :type IC: type[InstallationConfig]
    :param GC: `GameConfig` subclass type
    :type GC: type[GameConfig]
    :param _DGlobS: `DefaultGlobalSettings` subclass type. Example only.
    :type _DGlobS: type[DefaultGlobalSettings]
    :return: ConfigHolder with all configuration levels
    :rtype: ConfigHolder
    :raises RuntimeError: If requested game isn't registered
    """
    ICDSettings = IC._get_u()
    GCDSettings = GC._get_u()

    # Read global config.
    global_config = GlobalConfig.from_toml(config_path, _DGlobS)

    # Check if game exists.
    if project_name not in (project_names := global_config.settings.projects):
        raise RuntimeError(
            f'{project_name} not installed or not registered in {project_names}'
        )

    # Read installation config.
    installation_config_path = global_config.settings.data_dir.joinpath(
        project_name + '.toml'
    )
    installation_config = InstallationConfig.from_toml(
        installation_config_path, ICDSettings
    )

    # Read game config.
    game_config_path = installation_config.settings.data_game_config
    game_config = GameConfig.from_toml(game_config_path, GCDSettings)

    return ConfigHolder(global_config, installation_config, game_config)
