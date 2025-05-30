#!/usr/bin/env python3
from abc import ABC, abstractmethod
from pathlib import Path

from chromadb import QueryResult, PersistentClient
from chromadb.utils import embedding_functions
from config.config import ConfigHolder, GameConfig, InstallationConfig
from config.structure import GameSettings, InstallationSettings
from llm.sources import downloader
from util.logger import get_logger

logger = get_logger(__name__)

_db_name = 'chroma.sqlite3'


class DatabaseProvider(ABC):
    @abstractmethod
    def __init__(self, confh: ConfigHolder) -> None: ...

    @abstractmethod
    def query(
        self,
        domain_name: str,
        query_texts: list[str],
        n_results: int,
        where: dict | None,
        where_document: dict | None,
        ids: list[str] | None,
    ) -> QueryResult: ...


class Database:
    def __init__(
        self, provider: type[DatabaseProvider], confh: ConfigHolder
    ) -> None:
        self.provider = provider(confh)

    def query(
        self,
        domain_name: str,
        query_texts: list[str],
        n_results: int,
        where: dict[str, str] | None = None,
        where_document: dict[str, str] | None = None,
        ids: list[str] | None = None,
    ) -> QueryResult:
        logger.debug(f'Query {domain_name} with batch size {len(query_texts)}')
        return self.provider.query(
            domain_name, query_texts, n_results, where, where_document, ids
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.provider.__class__.__name__})'


class Chroma(DatabaseProvider):
    def __init__(self, confh: ConfigHolder) -> None:
        self.confh: ConfigHolder[
            InstallationConfig, GameConfig, InstallationSettings, GameSettings
        ] = confh
        logger.debug('Attempt to load chromadb database.')
        # Store some settings from `confh`.
        self.db_dir: Path = self.confh.inst.settings.database_db_dir
        self.em_dir: Path = self.confh.insts.database_embedding_models_dir
        self.model_name: str = self.confh.games.database_embedding_model_name

        # Download model. Always try to fix missing.
        # Use `self.em_dir` as the downloader cache dir.
        self.model_path: Path = downloader(
            repo=self.model_name,
            emb_dir=self.em_dir,
            cache_dir=self.em_dir,
            fix_missing=False,
        )

        # Load the embedding model.
        logger.debug(f'Embedding fuction is {self.model_name}.')
        self.model = embedding_functions.SentenceTransformerEmbeddingFunction(
            self.model_path.as_posix()
        )
        # Load the database.
        logger.debug(f'Persistent client at {self.db_dir.joinpath(_db_name)}.')
        self.client = PersistentClient(
            path=self.confh.insts.database_db_dir.as_posix()
        )
        logger.debug('Successfully loaded chromadb database.')

    def query(
        self,
        domain_name: str,
        query_texts: list[str],
        n_results: int,
        where: dict | None,
        where_document: dict | None,
        ids: list[str] | None,
    ) -> QueryResult:
        collection = self.client.get_collection(domain_name)
        return collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            ids=ids,
        )

    def __repr__(self) -> str:
        return f'<DatabaseProvider {self.__class__.__name__}>'
