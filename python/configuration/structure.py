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
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TypeAlias,
    TypeVar,
)

import xdg_base_dirs as xdg
from configuration.settings import DefaultSettings, Settings

V = TypeVar('V', bound='InstallationSettings')
W = TypeVar('W', bound='DefaultInstallationSettings')


X = TypeVar('X', bound='GameSettings')
Y = TypeVar('Y', bound='DefaultGameSettings')

A = TypeVar('A')

_sfx = '.toml'


def dfac(obj: A) -> A:
    return field(default_factory=lambda: obj)


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
    config_path: Path = config_dir.joinpath(prefix).with_suffix(_sfx)


@dataclass
class GlobalSettings(Settings):
    """
    Global application settings.

    :ivar data_dir: Base directory for application data.
    :ivar cache_dir: Base directory for cached files.
    :ivar projects: List of installed project identifiers.
    """

    # [data]
    data_dir: Path
    # [cache]
    cache_dir: Path
    # [projects]
    projects: list[str]


@dataclass
class DefaultGlobalSettings(DefaultSettings[GlobalSettings]):
    """Default values for global settings using XDG base directories."""

    # [data]
    data_dir: Path = xdg.xdg_data_home().joinpath(GlobalInfo.prefix)
    # [cache]
    cache_dir: Path = xdg.xdg_cache_home().joinpath(GlobalInfo.prefix)
    # [projects]
    projects: list[str] = dfac([])


GlobS: TypeAlias = GlobalSettings
DGlobS: TypeAlias = DefaultGlobalSettings


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
    :ivar project_data_dir: Project-specific data directory.
    :ivar project_cache_dir: Project-specific cache directory.
    :ivar data_game_dir: Directory for game data files.
    :ivar data_game_config: Path to game-specific configuration.
    :ivar cache_assets_dir: Directory for cached assets. Not modified.
    :ivar cache_session_dir: Directory for cached session. Modified.
    :ivar database_db_dir: Directory for databases.
    :ivar database_embedding_models_dir: Directory for embedding models.
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
    cache_assets_plot_dir: Path
    cache_assets_cc_dir: Path
    cache_session_dir: Path
    cache_session_plot_dir: Path
    cache_session_cc_dir: Path

    # [database]
    database_db_dir: Path
    database_embedding_models_dir: Path


@dataclass
class DefaultInstallationSettings(DefaultSettings[V]):
    """Default values for installation settings."""

    # [project]
    project_name: str = 'default_project'
    project_data_dir: Path = DGlobS.data_dir.joinpath(project_name)
    project_cache_dir: Path = DGlobS.cache_dir.joinpath(project_name)

    # [data]
    data_game_dir: Path = project_data_dir
    data_game_config: Path = data_game_dir.joinpath(project_name).with_suffix(
        _sfx
    )

    # [cache]
    cache_assets_dir: Path = project_cache_dir.joinpath('assets')
    cache_assets_plot_dir: Path = cache_assets_dir.joinpath('plot')
    cache_assets_cc_dir: Path = cache_assets_dir.joinpath('cc')
    cache_session_dir: Path = project_cache_dir.joinpath('session')
    cache_session_plot_dir: Path = cache_session_dir.joinpath('plot')
    cache_session_cc_dir: Path = cache_assets_dir.joinpath('cc')

    # [database]
    database_db_dir: Path = cache_assets_dir.joinpath('db')
    database_embedding_models_dir: Path = DGlobS.cache_dir.joinpath(
        'embedding_models'
    )


@dataclass
class GameSettings(Settings):
    """
    Base class for game-specific settings.

    :ivar database_embedding_model_name: Name of the embedding model.
    """

    # [database]
    database_embedding_model_name: str
    database_embedding_model_type: str
    database_embedding_model_instruction_aware: bool
    database_embedding_model_instruction_aware_fstring: str
    database_embedding_model_kwargs: dict[str, str]
    database_embedding_model_tokenizer_kwargs: dict[str, str]
    database_embedding_model_embedding_fn_kwargs: dict[
        str, bool | int | float | str
    ]
    database_reranker_model_name: str
    database_reranker_model_type: str
    database_reranker_model_instruction_aware: bool
    database_reranker_model_instruction_aware_fstring: str
    database_reranker_model_kwargs: dict[str, str]
    database_reranker_model_tokenizer_kwargs: dict[str, str]
    database_reranker_model_embedding_fn_kwargs: dict[
        str, bool | int | float | str
    ]


@dataclass
class DefaultGameSettings(DefaultSettings[X]):
    """Base class for default game-specific settings."""

    # [database]
    database_embedding_model_name: str = 'BAAI/bge-small-en-v1.5'
    database_embedding_model_type: str = 'sentence_transformers'
    database_embedding_model_instruction_aware: bool = False
    database_embedding_model_instruction_aware_fstring: str = ''
    database_embedding_model_kwargs: dict[str, str] = dfac({})
    database_embedding_model_tokenizer_kwargs: dict[str, str] = dfac({})
    database_embedding_model_embedding_fn_kwargs: dict[
        str, bool | int | float | str
    ] = dfac({})
    database_reranker_model_name: str = ''  # 'BAAI/bge-reranker-base'
    database_reranker_model_type: str = ''  # 'CustomHuggingFace'
    database_reranker_model_instruction_aware: bool = False
    database_reranker_model_instruction_aware_fstring: str = ''
    database_reranker_model_kwargs: dict[str, str] = dfac({})
    database_reranker_model_tokenizer_kwargs: dict[str, str] = dfac({})
    database_reranker_model_embedding_fn_kwargs: dict[
        str, bool | int | float | str
    ] = dfac({})
