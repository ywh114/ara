#!/usr/bin/env python3
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import run  # Call `tree` for the example only.

from util.config import (
    DefaultGameSettings as DGameS,
)
from util.config import (
    DefaultGlobalSettings as DGlobS,
)
from util.config import (
    DefaultInstallationSettings as DInstS,
)
from util.config import (
    GameConfig as GameC,
)
from util.config import (
    GameSettings as GameS,
)
from util.config import (
    GlobalInfo,
    loadconf,
)
from util.config import (
    InstallationConfig as InstC,
)
from util.config import (
    InstallationSettings as InstS,
)

######################
### Example setup. ###
######################
fakehome = Path.cwd().joinpath('__cache__')
pfx = GlobalInfo.prefix
project_name = 'example_project'


@dataclass
class ExampleDGlobS(DGlobS):
    data_dir: Path = fakehome.joinpath('local', 'share').joinpath(pfx)
    cache_dir: Path = fakehome.joinpath('local', 'cache').joinpath(pfx)
    projects: list[str] = field(default_factory=lambda: [project_name])


example_config_path = GlobalInfo.config_path
example_config_path = fakehome.joinpath('config', pfx + '.conf')  # Example only
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
# Required for all new projects.
@dataclass
class ExampleISet(InstS):
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
    embedding_model_name: str  # New field


@dataclass
class ExampleDISet(DInstS[ExampleISet]):
    # [project]
    project_name: str = 'example_project'
    project_data_dir: Path = ExampleDGlobS.data_dir.joinpath(project_name)
    project_cache_dir: Path = ExampleDGlobS.cache_dir.joinpath(project_name)

    # [data]
    data_game_dir: Path = project_data_dir.joinpath('game')
    data_game_config: Path = project_data_dir.joinpath(f'{project_name}.toml')

    # [cache]
    cache_assets_dir: Path = project_cache_dir.joinpath('assets')

    # [embedding]
    embedding_db_dir: Path = cache_assets_dir.joinpath('db')
    embedding_models_dir: Path = ExampleDGlobS.cache_dir.joinpath(
        'embedding_models'
    )
    embedding_model_name: str = 'BAAI/bge-small-en-v1.5'  # New field


@dataclass
class ExampleGSet(GameS):
    dummy_int: int


@dataclass
class ExampleDGSet(DGameS[ExampleGSet]):
    dummy_int: int = 2


#######################
### Custom configs. ###
#######################
# Required for all new projects.
class ExampleIConf(InstC[ExampleISet, ExampleDISet]): ...


class ExampleGConf(GameC[ExampleGSet, ExampleDGSet]): ...


#####################
### Load configs. ###
#####################
ch = loadconf(
    example_config_path,
    project_name,
    ExampleIConf,
    ExampleGConf,
    ExampleDGlobS,
)

ch.glob.write_toml()
ch.inst.write_toml()
ch.game.write_toml()
print(f'Initialized `{project_name}` under `{fakehome}`.')
print('\n------------------------------------')
input(f"tree '{fakehome}' (enter to run) ")
run(('tree', fakehome))
print('------------------------------------')

print('\n------------------------------------')
input(f"$ cat '{ch.glob.path}' (enter to run) ")
run(('cat', ch.glob.path))
print('------------------------------------')

print('\n------------------------------------')
input(f"$ cat '{ch.inst.path}' (enter to run) ")
run(('cat', ch.inst.path))
print('------------------------------------')

print('\n------------------------------------')
input(f"$ cat '{ch.game.path}' (enter to run) ")
run(('cat', ch.game.path))
print('------------------------------------')
