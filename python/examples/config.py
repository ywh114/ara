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
from dataclasses import field
from pathlib import Path
from subprocess import run  # For the example only.

from configuration import (
    ConfigHolder,
    DefaultGlobalSettings,
    GlobalInfo,
    dfac,
    load_conf3,
)
from configuration import DefaultGameSettings as DGSettings
from configuration import DefaultInstallationSettings as DISettings
from configuration import GameConfig as GConfig
from configuration import GameSettings as GSettings
from configuration import InstallationConfig as IConfig
from configuration import InstallationSettings as ISettings
from pydantic.dataclasses import dataclass
from utils.logger import get_logger

# NOTE: Use pydantic for verification when loading from files.

logger = get_logger(__name__)

######################
### Example setup. ###
######################
example_assets = Path.cwd().joinpath('examples', 'assets')
fakehome = example_assets.joinpath('__cache__')
pfx = GlobalInfo.prefix
project_name = 'example_project'


@dataclass
class _DGS(DefaultGlobalSettings):
    data_dir: Path = fakehome.joinpath('local', 'share').joinpath(pfx)
    cache_dir: Path = fakehome.joinpath('cache').joinpath(pfx)
    projects: list[str] = field(default_factory=lambda: [project_name])


example_config_path = GlobalInfo.config_path
example_config_path = fakehome.joinpath(
    'config', pfx, f'{pfx}.toml'
)  # Example only
#####################
### Ignore above. ###
#####################


# From util.config:
# NOTE: Subclass from installation and game settings. Avoid modifying any global
# settings unless for testing/an example; use `DefaultGlobalSettings`.
# XXX: All fields must be manually filled in when subclassing. There is no
# automatic concatanation of paths.


########################
### Custom settings. ###
########################
# Always required.
@dataclass
class ExampleISettings(ISettings):
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
class ExampleDISettings(DISettings[ExampleISettings]):
    # [project]
    project_name: str = 'example_project'
    project_data_dir: Path = _DGS.data_dir.joinpath(project_name)
    project_cache_dir: Path = _DGS.cache_dir.joinpath(project_name)

    # [data]
    data_game_dir: Path = project_data_dir
    data_game_config: Path = data_game_dir.joinpath(f'{project_name}.toml')

    # [cache]
    cache_assets_dir: Path = project_cache_dir.joinpath('assets')
    cache_assets_plot_dir: Path = cache_assets_dir.joinpath('plot')
    cache_assets_cc_dir: Path = cache_assets_dir.joinpath('cc')
    cache_session_dir: Path = project_cache_dir.joinpath('session')
    cache_session_plot_dir: Path = cache_session_dir.joinpath('plot')
    cache_session_cc_dir: Path = cache_session_dir.joinpath('cc')

    # [database]
    database_db_dir: Path = cache_session_dir.joinpath('db')
    database_embedding_models_dir: Path = _DGS.cache_dir.joinpath(
        'embedding_models'
    )


@dataclass
class ExampleGSettings(GSettings):
    # [api]
    api_example_api_key_env_var: str
    api_example_api_endpoint: str
    api_example_api_model: str
    # [database]
    database_embedding_model_name: str
    database_embedding_model_type: str
    database_embedding_model_kwargs: dict[str, str]
    database_embedding_model_instruction_aware: bool
    database_embedding_model_instruction_aware_fstring: str
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
    # [dummy]
    dummy_init: bool


# NOTE: To use flash attention:
#   database_embedding_model_kwargs: dict[str, str] = dfac(
#       {
#           'attn_implementation': 'flash_attention_2',
#           'device_map': 'auto',
#       }
#   )
#   database_embedding_model_tokenizer_kwargs: dict[str, str] = dfac(
#       {'padding_side': 'left'}
#   )
#   database_embedding_model_embedding_fn_kwargs = dfac(
#       {'device': 'cuda'}
#   )
@dataclass
class ExampleDGSettings(DGSettings[ExampleGSettings]):
    # [api]
    api_example_api_key_env_var: str = 'DEEPSEEK_API_KEY'
    api_example_api_endpoint: str = 'https://api.deepseek.com'
    api_example_api_model: str = 'deepseek-chat'
    # [database]
    database_embedding_model_name: str = (
        'BAAI/bge-small-en-v1.5'  # 'Qwen/Qwen3-Embedding-0.6B'
    )
    database_embedding_model_type: str = 'SentenceTransformer'
    database_embedding_model_instruction_aware: bool = True
    database_embedding_model_instruction_aware_fstring: str = (
        'Instruct: {task}\nQuery: {query}'  # Use {task} and {query} only.
    )
    database_embedding_model_kwargs: dict[str, str] = dfac({})
    database_embedding_model_tokenizer_kwargs: dict[str, str] = dfac({})
    database_embedding_model_embedding_fn_kwargs: dict[
        str, bool | int | float | str
    ] = dfac({})
    database_reranker_model_name: str = ''  #'Qwen/Qwen3-Reranker-0.6B'
    database_reranker_model_type: str = ''  #'CustomHuggingFace4Qwen3'
    database_reranker_model_instruction_aware: bool = True
    database_reranker_model_instruction_aware_fstring: str = (
        '<Instruct>: {task}\n<Query>: {query}'  # Use {task} and {query} only.
    )
    database_reranker_model_kwargs: dict[str, str] = dfac({})
    database_reranker_model_tokenizer_kwargs: dict[str, str] = dfac({})
    database_reranker_model_embedding_fn_kwargs: dict[
        str, bool | int | float | str
    ] = dfac({'device': 'cuda'})  # Give GPU to expensive reranker.
    # [dummy]
    dummy_init: bool = False


#######################
### Custom configs. ###
#######################
# Always required.
class ExampleIConfig(IConfig[ExampleISettings, ExampleDISettings]): ...


class ExampleGConfig(GConfig[ExampleGSettings, ExampleDGSettings]): ...


#####################
### Load configs. ###
#####################
def initialize_and_show(confh: ConfigHolder) -> None:
    def logrun(*args: str | Path) -> None:
        result = run(args, capture_output=True, text=True)
        if result.stderr:
            logger.error(result.stderr)

        logger.info(result.stdout)

    logger.info(f'Initialized `{project_name}` under `{fakehome}`.')
    input(f"tree '{fakehome}' (enter to run) ")
    logrun('tree', fakehome)

    logger.info('------------------------------------')

    input(f"$ cat '{confh.glob.path}' (enter to run) ")
    logrun('cat', confh.glob.path)
    logger.info('------------------------------------')

    input(f"$ cat '{confh.inst.path}' (enter to run) ")
    logrun('cat', confh.inst.path)

    logger.info('------------------------------------')

    input(f"$ cat '{confh.game.path}' (enter to run) ")
    logrun('cat', confh.game.path)


confh: ConfigHolder[
    ExampleIConfig, ExampleGConfig, ExampleISettings, ExampleGSettings
] = load_conf3(
    example_config_path,  # Example only. Use `GlobalInfo.config_path`.
    project_name,
    ExampleIConfig,
    ExampleGConfig,
    _DGS=_DGS,  # Set _DefaultGlobalSettings for example only.
)

if __name__ == '__main__':
    initialize_and_show(confh)
