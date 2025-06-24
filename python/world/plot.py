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
from typing import Iterator, Self

import tomli_w
from configuration import ConfigHolder, P, Q, V, X
from configuration.plot import (
    DefaultPlotSettings,
    PlotConfig,
    PlotSettings,
)
from llm.database import DatabaseProvider
from llm.utils.context_manager import Context, MultiroundContextManager
from utils.logger import get_logger
from utils.uuid4_from_seed import uuid4_from_seed
from world.character.character_class import standard_load

logger = get_logger(__name__)

_sfx = '.toml'


# XXX: Location names must be unique throughout the entire game.
class Location:
    domain = 'locations'
    desc_salt = '_desc'
    lore_salt = '_lore'

    def __init__(
        self, db: DatabaseProvider, name: str, desc: str = '', lore: str = ''
    ) -> None:
        self.db = db
        self.name = name
        self.desc = desc
        self.lore = lore

        self.desc_id = str(uuid4_from_seed(self.name + self.desc_salt))
        self.lore_id = str(uuid4_from_seed(self.name + self.lore_salt))

        self.db.touch(self.domain)  # Touch the collection.
        gr = self.db.get(
            self.domain,
            where=None,
            where_document=None,
            ids=[self.desc_id, self.lore_id],
        )

        # Load from db.
        if gr and (ids := gr['ids']) and (docs := gr['documents']):
            if self.desc_id in ids:
                logger.debug(f'Read description of {self} from {db}')
                self.desc = docs[ids.index(self.desc_id)]
            if self.lore_id in ids:
                self.lore = docs[ids.index(self.lore_id)]
                logger.debug(f'Read lore of {self} from {db}')

        logger.debug(f'Update {self} in {db}.')
        self.update()

    # Update in db
    def update(self) -> None:
        self.db.upsert(
            self.domain,
            [self.desc_id, self.lore_id],
            documents=[self.desc, self.lore],
        )

    @property
    def scratch(self) -> Context:
        return [
            {
                'role': 'user',
                'content': 'The following is a description of your '
                'surroundings.',
                'name': MultiroundContextManager.default_sysname,
            },
            {
                'role': 'assistant',
                'content': 'Description:\n'
                + self.desc
                + '\nLore:\n'
                + self.lore,
                'name': MultiroundContextManager.default_sysname,
            },
        ]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object, /) -> bool:
        assert isinstance(other, Location)
        return self.name == other.name

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.name}>'


class Plot:  # TODO: Implement
    breadcrumbs_key = 'breadcrumbs'

    def __init__(
        self,
        db: DatabaseProvider,
        confh: ConfigHolder[P, Q, V, X],
        plots: PlotSettings,
        prev: Self | None = None,
    ) -> None:
        self.db = db
        self.confh = confh
        # Misc
        self.id = plots.id
        self.language = plots.language
        self.zeitgeist = plots.zeitgeist
        self.tone = plots.tone
        # Meta
        self.prev = prev
        self.id_chain = [self.id]
        if self.prev is not None:
            self.id_chain = self.prev.id_chain + [self.id]
        confh.insts.cache_session_plot_dir.mkdir(parents=True, exist_ok=True)
        with open(
            confh.insts.cache_session_plot_dir.joinpath(
                self.breadcrumbs_key
            ).with_suffix(_sfx),
            'wb+',
        ) as f:
            tomli_w.dump({self.breadcrumbs_key: self.id_chain}, f)

        # Characters
        self.character_pool = set(
            standard_load(name, db, confh) for name in plots.character_pool
        )
        logger.debug(f'Character pool: {self.character_pool}')

        self.starting_characters = set(
            char
            for char in self.character_pool
            if char.name in plots.character_inits
        )
        logger.debug(f'Starting characters: {self.starting_characters}')

        if _x := self.starting_characters - self.character_pool:
            raise RuntimeError(f'{_x} not in pool {self.character_pool}.')

        # Locations
        self.location_pool = set(
            Location(db, name, plots.location_descs[name])
            for name in plots.location_pool
        )
        logger.debug(f'Location pool: {self.location_pool}')

        try:
            self.starting_location = next(
                loc
                for loc in self.location_pool
                if loc.name == plots.location_init
            )
        except StopIteration as e:
            if not hasattr(self, 'starting_location'):
                raise RuntimeError(f'Failed to load plot {self}.') from e
            else:
                raise RuntimeError(
                    f'{self.starting_location} not in pool {self.location_pool}.'
                ) from e
        logger.debug(f'Starting location: {self.starting_location}')

        # Plot
        self.plot_considerations = plots.plot_considerations
        self.plot_story = plots.plot_story
        self.plot_next_considerations = plots.plot_next.pop(
            'considerations', 'None'
        )
        self.plot_next = plots.plot_next

        is_only = any(
            self.prev_id in details.get('only_for', [])
            for details in self.plot_next.values()
        )
        # If `name` is in `only_for`, only choices are options marked `only_for`
        # with the `name`. Otherwise, choices are unmarked options.
        # TODO: Write TypedDict
        if is_only:
            self.plot_next_filtered: dict[str, str] = {  # pyright: ignore [reportAttributeAccessIssue]
                name: details.get('desc', '')
                for name, details in self.plot_next.items()
                if name in details.get('only_for', [])
            }
        else:
            self.plot_next_filtered = {  # pyright: ignore [reportAttributeAccessIssue]
                name: details['desc']
                for name, details in self.plot_next.items()
                if not details.get('only_for', [])
            }
        self.plot_next_valid_choices = set(self.plot_next_filtered.keys())

        logger.debug(f'Finished initialization of plot {self}.')

    @classmethod
    def new(
        cls,
        db: DatabaseProvider,
        confh: ConfigHolder[P, Q, V, X],
        name: str,
        prev: Self | None = None,
    ) -> Self:
        path = confh.insts.cache_assets_plot_dir.joinpath(name).with_suffix(
            _sfx
        )
        try:
            plotc = PlotConfig.load_toml(path, DefaultPlotSettings)
        except Exception as e:
            raise RuntimeError(f'Failed to load plot file {path}: {e}') from e

        # Sanitize prev.
        if prev is not None:
            prev.prev = None

        return cls(
            db,
            confh,
            plotc.settings,
            prev,
        )

    @property
    def prev_id(self) -> str:
        if self.id_chain:
            return self.id_chain[-1]
        else:
            return ''

    @property
    def next_choices_pretty(self) -> str:
        return '\n'.join(
            f'{name}: {desc}' for name, desc in self.plot_next_filtered.items()
        )

    @property
    def as_tool_contents(self) -> str:
        # TODO:
        return (
            f'Considerations:\n{self.plot_considerations}\n'
            f'Plot:\n{self.plot_story}\n'
            f'Next scene choices:\n{self.next_choices_pretty}'
        )

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.id}>'


class PlotMarcher(Iterator[Plot]):
    def __init__(self, plot: Plot) -> None:
        self.plot = plot
        self.next = None
        self.valid = self.plot.plot_next_valid_choices

    def elect(self, next: str) -> None:
        if next in self.valid:
            self.next = next
        else:
            raise RuntimeError(f'{next} not in {self.valid}')

    def __next__(self) -> Plot:
        if self.next is None:
            raise StopIteration
        else:
            return Plot.new(
                db=self.plot.db,
                confh=self.plot.confh,
                name=self.next,
                prev=self.plot,
            )
