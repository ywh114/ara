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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, override

from chromadb import (
    Documents,
    GetResult,
    IDs,
    Metadata,
    Metadatas,
    PersistentClient,
    QueryResult,
    Where,
    WhereDocument,
)
from chromadb.api.types import ID, Document, Embedding, OneOrMany
from chromadb.utils import embedding_functions as ef
from configuration.config import ConfigHolder, GameConfig, InstallationConfig
from configuration.structure import GameSettings, InstallationSettings

# from llm.utils.reranker import (
#    CustomHuggingFace4Qwen3RerankerFunction,
#    CustomHuggingFaceRerankerFunction,
# )
from llm.utils.sources import downloader
from utils.logger import get_logger

logger = get_logger(__name__)

_db_name = 'chroma.sqlite3'


class DatabaseProvider(ABC):
    """Abstract base class for database providers."""

    @abstractmethod
    def __init__(self, confh: ConfigHolder) -> None: ...

    @abstractmethod
    def query(
        self,
        domain_name: str,
        query_texts: list[str],
        n_results: int,
        where: Where | None,
        where_document: WhereDocument | None,
        ids: IDs | None,
        instruct_task: str = '',
        rerank_task: str = '',
        with_reranker: bool = False,
        reranker_context: bool = False,
    ) -> QueryResult:
        """
        Execute a query against the database.

        :param domain_name: Collection name to query against.
        :param query_texts: List of text queries to execute.
        :param n_results: Number of results to return per query.
        :param where: Metadata filter conditions.
        :param where_document: Document content filter conditions.
        :param ids: Specific document IDs to include in results.
        :param instruct_task: Instruct task for instruction-aware embedding
        models or if using a reranker.
        :param with_reranker: Enable reranking.
        :param reranker_context: Give context to the reranker.
        :return: Query results with documents and metadata.
        """

    @abstractmethod
    def get(
        self,
        domain_name: str,
        where: Where | None,
        where_document: WhereDocument | None,
        ids: IDs | None,
    ) -> GetResult:
        """
        Execute a get against the database.
        :param domain_name: Target collection name.
        :param where: Metadata filter conditions.
        :param where_document: Document content filter conditions.
        :param ids: Specific document IDs to include in results.
        :return: Get results with documents and metadata.
        """

    @abstractmethod
    def add(
        self,
        domain_name: str,
        documents: Documents,
        metadatas: Metadatas,
        ids: IDs,
    ) -> None:
        """
        Add documents to a collection.

        :param domain_name: Target collection name.
        :param documents: Documents to add.
        :param metadatas: Metadata for each document.
        :param ids: Unique IDs for each document.
        """

    @abstractmethod
    def upsert(
        self,
        domain_name: str,
        ids: OneOrMany[ID],
        embeddings: OneOrMany[Embedding] | None = None,
        metadatas: OneOrMany[Metadata] | None = None,
        documents: OneOrMany[Document] | None = None,
    ) -> None:
        """
        Upsert documents.
        """

    @abstractmethod
    def touch(self, domain_name: str) -> None:
        """
        Touch a collection.

        :param domain_name: Target collection name.
        """


class Chroma(DatabaseProvider):
    """
    ChromaDB. Install packages as needed.
    ChromaDB database provider implementation.

    Attributes:
        ef_dict: Supported embedding function types.
        rf_dict: Supported reranker function types.
        client: Persistent ChromaDB client.
        model: Active embedding model.
        rmodel: Active reranker model.

    See https://docs.trychroma.com/docs/overview/introduction.
    """

    ef_dict = {
        'Cohere': ef.CohereEmbeddingFunction,
        'OpenAI': ef.OpenAIEmbeddingFunction,
        'HuggingFace': ef.HuggingFaceEmbeddingFunction,
        'SentenceTransformer': ef.SentenceTransformerEmbeddingFunction,
        'GooglePalm': ef.GooglePalmEmbeddingFunction,
        'GoogleGenerativeAi': ef.GoogleGenerativeAiEmbeddingFunction,
        'GoogleVertex': ef.GoogleVertexEmbeddingFunction,
        'Ollama': ef.OllamaEmbeddingFunction,
        'Instructor': ef.InstructorEmbeddingFunction,
        'Jina': ef.JinaEmbeddingFunction,
        'Mistral': ef.MistralEmbeddingFunction,
        'VoyageAI': ef.VoyageAIEmbeddingFunction,
        'ONNXMiniLM_L6_V2': ef.ONNXMiniLM_L6_V2,
        'OpenCLIP': ef.OpenCLIPEmbeddingFunction,
        'Roboflow': ef.RoboflowEmbeddingFunction,
        'Text2Vec': ef.Text2VecEmbeddingFunction,
        'AmazonBedrock': ef.AmazonBedrockEmbeddingFunction,
        'ChromaLangchain': ef.ChromaLangchainEmbeddingFunction,
        'Baseten': ef.BasetenEmbeddingFunction,
        'CloudflareWorkersAI': ef.CloudflareWorkersAIEmbeddingFunction,
        'TogetherAI': ef.TogetherAIEmbeddingFunction,
        'Default': ef.DefaultEmbeddingFunction,
    }
    rf_dict = {}
    # rf_dict = {
    #    'CustomHuggingFace': CustomHuggingFaceRerankerFunction,
    #    'CustomHuggingFace4Qwen3': CustomHuggingFace4Qwen3RerankerFunction,
    # }

    @override
    def __init__(self, confh: ConfigHolder) -> None:
        """
        Initialize ChromaDB provider with configuration.

        :param confh: Configuration holder instance.
        :raises RuntimeError: For unsupported model types.
        :raises ValueError: For invalid model initialization.
        """
        self.confh: ConfigHolder[
            InstallationConfig, GameConfig, InstallationSettings, GameSettings
        ] = confh
        logger.debug('Attempt to load chromadb database.')
        # Store some settings from `confh`.
        self.db_dir: Path = self.confh.inst.settings.database_db_dir
        self.em_dir: Path = self.confh.insts.database_embedding_models_dir
        # Embedding model.
        self.model_name: str = self.confh.games.database_embedding_model_name
        self.model_type: str = self.confh.games.database_embedding_model_type
        # Reranker model.
        self.rmodel_name: str = self.confh.games.database_reranker_model_name
        self.rmodel_type: str = self.confh.games.database_reranker_model_type
        # Embedding model instruct options.
        self.instruct_aware = (
            self.confh.games.database_embedding_model_instruction_aware
        )
        self.instruct_fstring = (
            self.confh.games.database_embedding_model_instruction_aware_fstring
        )
        self.reranker_instruct_aware = (
            self.confh.games.database_reranker_model_instruction_aware
        )
        self.reranker_fstring = (
            self.confh.games.database_reranker_model_instruction_aware_fstring
        )

        # Embedding and reranker model kwargs.
        # XXX: {'attn_implementation': 'flash_attention_2'} requires the
        # `flash-attn` package.
        self.model_kwargs: dict[str, str] = (
            self.confh.games.database_embedding_model_kwargs
        )
        self.model_tokenizer_kwargs: dict[str, str] = (
            self.confh.games.database_embedding_model_tokenizer_kwargs
        )
        self.embedding_fn_kwargs: dict[str, str | int | float | bool] = (
            self.confh.games.database_embedding_model_embedding_fn_kwargs
        )
        self.rmodel_kwargs: dict[str, str] = (
            self.confh.games.database_reranker_model_kwargs
        )
        self.rmodel_tokenizer_kwargs: dict[str, str] = (
            self.confh.games.database_reranker_model_tokenizer_kwargs
        )
        self.rembedding_fn_kwargs: dict[str, str | int | float | bool] = (
            self.confh.games.database_reranker_model_embedding_fn_kwargs
        )
        # Log kwargs debug.
        logger.debug(f'Embedding model kwargs: {self.model_kwargs}')
        logger.debug(
            f'Embedding model tokenizer kwargs: {self.model_tokenizer_kwargs}'
        )
        logger.debug(f'Embedding function kwargs: {self.embedding_fn_kwargs}')
        logger.debug(f'Embedding model instruct-aware: {self.instruct_aware}')
        logger.debug(f'Instruction fstring: {self.instruct_fstring}')
        logger.debug(f'Reranker model kwargs: {self.rmodel_kwargs}')
        logger.debug(
            f'Reranker model tokenizer kwargs: {self.rmodel_tokenizer_kwargs}'
        )
        logger.debug(f'Reranker function kwargs: {self.rembedding_fn_kwargs}')

        if self.model_type not in self.ef_dict:
            raise RuntimeError(
                f'{self.model_type} not supported: {tuple(self.ef_dict.keys())}'
            )
        if self.rmodel_type and self.rmodel_type not in self.rf_dict:
            raise RuntimeError(
                f'{self.rmodel_type} not supported: {tuple(self.rf_dict.keys())}'
            )

        logger.debug(f'Embedding fuction type is {self.model_type}.')
        logger.debug(f'Reranker fuction type is {self.rmodel_type}.')

        # Download models. Always try to fix missing.
        # Use `self.em_dir` as the downloader cache dir.
        self.model_path: Path = downloader(
            repo=self.model_name,
            emb_dir=self.em_dir,
            cache_dir=self.em_dir.joinpath('.cache'),
            fix_missing=not __debug__,
        )
        if self.rmodel_name:
            self.rmodel_path: Path = downloader(
                repo=self.rmodel_name,
                emb_dir=self.em_dir,
                cache_dir=self.em_dir.joinpath('.cache'),
                fix_missing=not __debug__,
            )

        # Load the embedding model.
        logger.debug(f'Embedding fuction is {self.model_name}.')
        try:
            self.model = self.ef_dict[self.model_type](
                self.model_path.as_posix(),
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.model_tokenizer_kwargs,
                **self.embedding_fn_kwargs,
            )
        except ValueError as e:
            raise ValueError(
                'Only sentence_transformers comes pre-installed. '
                f'Please follow the instructions: {e}'
            ) from e
        # Load the reranker model.
        if self.rmodel_name:
            logger.debug(f'Reranker fuction is {self.rmodel_name}.')
            try:
                self.rmodel = self.rf_dict[self.rmodel_type](
                    self.rmodel_path.as_posix(),
                    model_kwargs=self.rmodel_kwargs,
                    tokenizer_kwargs=self.rmodel_tokenizer_kwargs,
                    **self.rembedding_fn_kwargs,
                )
            except ValueError as e:
                raise ValueError(f'Could not load reranker model: {e}') from e
        # Load the database.
        logger.debug(f'Persistent client at {self.db_dir.joinpath(_db_name)}.')
        self.client = PersistentClient(
            path=self.confh.insts.database_db_dir.as_posix()
        )
        logger.debug('Successfully loaded chromadb database.')

    @override
    def query(
        self,
        domain_name: str,
        query_texts: list[str],
        n_results: int,
        where: Where | None,
        where_document: WhereDocument | None,
        ids: IDs | None,
        instruct_task: str = '',
        rerank_task: str = '',
        with_reranker: bool = False,
        reranker_context: bool = False,
    ) -> QueryResult:
        """
        Execute a query against the database.

        :param domain_name: Collection name to query against.
        :param query_texts: List of text queries to execute.
        :param n_results: Number of results to return per query.
        :param where: Metadata filter conditions.
        :param where_document: Document content filter conditions.
        :param ids: Specific document IDs to include in results.
        :param instruct_task: Instruct task for instruction-aware embedding
        models or if using a reranker.
        :param with_reranker: Enable reranking.
        :param reranker_context: Give context to the reranker.
        :return: Query results with documents and metadata.
        """
        logger.debug(
            f'{self}: search on {domain_name} with size {len(query_texts)}.'
        )
        collection = self.client.get_collection(domain_name)

        query_result = collection.query(
            query_texts=self.format_embedding_instructs(
                task=instruct_task,
                query_texts=query_texts,
            ),
            n_results=n_results,
            where=where,
            where_document=where_document,
            ids=ids,
        )

        if with_reranker and self.rmodel_name:
            if (query_result_docs := query_result['documents']) is not None:
                logger.debug(
                    f'{self}: reranking documents in '
                    f'{len(query_result_docs)} batches.'
                )
                new_distances = [
                    self.rmodel(reranker_instruct)
                    for reranker_instruct in self.format_reranker_instructs(
                        task=rerank_task,
                        query_texts=query_texts,
                        query_result_docs=query_result_docs,
                        reranker_context=reranker_context,
                    )
                ]

                # Reorder based on new distances.
                def _transpose(r: Iterable[tuple[Any, Any]]) -> tuple:
                    return tuple(zip(*r))

                try:
                    query_result['documents'], query_result['distances'] = (
                        _transpose(
                            _transpose(
                                sorted(
                                    zip(query_result_doc, new_distance),
                                    reverse=True,
                                    key=lambda s: s[1],
                                )
                            )
                            for query_result_doc, new_distance in zip(
                                query_result_docs, new_distances
                            )
                        )
                    )
                except ValueError:
                    # Nothing found.
                    query_result['documents'] = []
                    query_result['distances'] = []

        return query_result

    @override
    def get(
        self,
        domain_name: str,
        where: Where | None,
        where_document: WhereDocument | None,
        ids: IDs | None,
    ) -> GetResult:
        logger.debug(f'{self}: get on {domain_name}.')
        collection = self.client.get_collection(domain_name)

        query_result = collection.get(
            where=where,
            where_document=where_document,
            ids=ids,
        )
        return query_result

    def format_embedding_instructs(
        self, task: str, query_texts: Documents
    ) -> list[str]:
        """
        Format queries for instruction-aware embedding models.

        :param task: Instruction task context.
        :param query_texts: Original query texts.
        :return: Formatted instruction-aware queries.
        """
        if self.instruct_aware:
            return [
                self.instruct_fstring.format(task=task, query=query)
                for query in query_texts
            ]
        else:
            return query_texts

    def format_reranker_instructs(
        self,
        task: str,
        query_texts: Documents,
        query_result_docs: list[Documents],
        reranker_context: bool = False,  # Only for extremely short entries.
    ) -> list[list[tuple[str, str]]]:
        """
        Format context for reranker model evaluation.

        Combines:
        - Instruction task
          - Instruct aware: Uses `...reranker_model_instruction_aware_fstring`.
            - Context: Adds other retrieved docs.
        - Original queries

        :param task: Reranking instruction task.
        :param query_texts: Original query texts.
        :param query_result_docs: Retrieved documents per query.
        :param reranker_context: Include context. Untested.
        :return: Formatted reranker inputs.
        """
        if not task:
            task = 'Retrieve passages relevant to the query.'
        if reranker_context and self.reranker_instruct_aware:
            task += '\nContext:\n{context}'

        def reformat(
            query_text: Document,
            doc: Document,
            docs: Documents,
        ) -> tuple[str, str]:
            if self.reranker_instruct_aware:
                return self.reranker_fstring.format(
                    task=task.format(context='\n'.join(docs)),
                    query=query_text,
                ), doc
            else:
                return query_text, doc

        return [
            [reformat(qt, doc, docs) for doc in docs]
            for qt, docs in zip(query_texts, query_result_docs)
        ]

    @override
    def add(
        self,
        domain_name: str,
        documents: Documents,
        metadatas: Metadatas,
        ids: IDs,
    ) -> None:
        """
        Add documents to a collection.

        :param domain_name: Target collection name.
        :param documents: Documents to add.
        :param metadatas: Metadata for each document.
        :param ids: Unique IDs for each document.
        """
        collection = self.client.get_or_create_collection(
            domain_name, embedding_function=self.model
        )
        return collection.add(documents=documents, metadatas=metadatas, ids=ids)

    @override
    def upsert(
        self,
        domain_name: str,
        ids: OneOrMany[ID],
        embeddings: OneOrMany[Embedding] | None = None,
        metadatas: OneOrMany[Metadata] | None = None,
        documents: OneOrMany[Document] | None = None,
    ) -> None:
        logger.debug(f'{self}: upsert {len(ids)} documents to {domain_name}.')
        collection = self.client.get_collection(domain_name)

        query_result = collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        return query_result

    @override
    def touch(self, domain_name: str) -> None:
        try:
            self.client.get_or_create_collection(
                domain_name, embedding_function=self.model
            )
        except ValueError as e:
            logger.warning(e)

    def __repr__(self) -> str:
        return f'<DatabaseProvider {self.__class__.__name__}>'
