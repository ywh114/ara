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
# https://cookbook.chromadb.dev/core/filters/#less-than-or-equal-lte
# TODO: Rewrite
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Mapping,
    Self,
    TypeAlias,
    TypeVar,
    override,
)

from chromadb import (
    Documents,
    IDs,
    Metadata,
    Metadatas,
    QueryResult,
    Where,
    WhereDocument,
)
from chromadb.api.types import ID, Document
from llm.database import DatabaseProvider
from utils.exceptions import KnowledgeSpecError
from utils.logger import get_logger
from utils.timestamp import timestamp

T = TypeVar('T', bound='KSearchSpec')
U = TypeVar('U', bound='KReturnSpec')

V = TypeVar('V', bound='KSpecExtender')

logger = get_logger(__name__)


@dataclass
class KSpecExtender:
    """
    Base class for knowledge specification extenders.
    Automatically creates a dictionary mapping of field names to values (with
    pluralized keys) during initialization.

    :ivar handle: Unique identifier for the extender instance.
    :ivar _dict: Automatically generated mapping of field names to values (with
        pluralized keys) excluding private fields.
    """

    handle: Any

    def __post_init__(self) -> None:
        """Initialize the `_dict` attribute after dataclass construction."""
        self._dict = {
            f'{name}s': getattr(self, name)
            for x in fields(self)
            if not (name := x.name).startswith('_')
        }


@dataclass
class KSearchSpecExtender(KSpecExtender):
    """
    Extender for search specifications.

    :ivar instruction: Query instruction for the search.
    :ivar min_or_max_distance: Distance threshold for result filtering.
    """

    instruction: str
    min_or_max_distance: float


@dataclass
class KAddSpecExtender(KSpecExtender):
    """
    Extender for addition specifications.

    :ivar document: Document content to add.
    :ivar metadata: Metadata associated with the document.
    :ivar id: Unique identifier for the document.
    """

    document: Document
    metadata: Metadata
    id: ID

    def __post_init__(self) -> None:
        metadata = {k: v for k, v in self.metadata.items()}
        metadata.setdefault('timestamp', timestamp.timestamp)
        self.metadata = metadata
        return super().__post_init__()


@dataclass
class NotExtendable(KSpecExtender):
    """Placeholder class for non-extendable specifications."""


KSSExtender: TypeAlias = KSearchSpecExtender
KASExtender: TypeAlias = KAddSpecExtender


@dataclass
class KSpecBase(ABC, Generic[V]):
    """
    Abstract base class for knowledge specifications.
    Provides extension capabilities for handling multiple specifications.
    Validates extender fields during initialization.

    :ivar handles: List of unique identifiers for extended specifications.
    """

    handles: list[Any]

    def __post_init__(self) -> None:
        """Validate extender fields against specification fields."""
        flds = {fld.name for fld in fields(self)}
        oflds = {
            f'{name}s'
            for fld in fields(self._get_x())
            if not (name := fld.name).startswith('_')
        }
        if delta := oflds - flds:
            raise KnowledgeSpecError(
                f'Extra {self._get_x()} field(s) for {self.__class__}: '
                f'{tuple(d[:-1] for d in delta)}'
            )

    @classmethod
    @abstractmethod
    def new(cls, *args, **kwargs) -> Self:
        """Abstract factory method to create new specifications."""

    @classmethod
    def _get_x(cls) -> type[V]:
        """
        Extract extender type from generic parameter.

        :return: The extender class type.
        :rtype: type[V]
        :raises SpecError: If constraints not followed.
        """
        for base in getattr(cls, '__orig_bases__', ()):
            if hasattr(base, '__args__') and base.__args__:
                arg = base.__args__[0]
                if not isinstance(arg, TypeVar):
                    return arg

        raise KnowledgeSpecError('Do not use bare `KSpecBase`.')

    def get_handle(self, handle: Any) -> int:
        """
        Get index position of a handle.

        :param handle: Handle to locate.
        :return: Index of the handle in the handles list.
        :raises ValueError: If handle not found.
        """
        try:
            return self.handles.index(handle)
        except ValueError as e:
            raise ValueError(
                f'Handle {handle} not registered: {self.handles}.'
            ) from e

    def pop_handle(self, handle: Any) -> V:
        """Remove and return specification associated with a handle.

        :param handle: Handle to remove.
        :return: Extender instance containing the removed specification.
        """
        k = self.get_handle(handle)

        return self._get_x()(
            **{
                name[:-1]: getattr(self, name).pop(k)
                for fld in fields(self)
                if not (name := fld.name).startswith('_')
            }
        )

    def extend(self, extender: V) -> None:
        """
        Extend specification with a new extender.

        :param extender: Extender instance to add.
        :raises KnowledgeSpecError: For invalid extender types.
        """
        if not isinstance(extender, self._get_x()):
            raise KnowledgeSpecError(
                f'Wrong spec: expected {self._get_x()}, not {type(extender)}.'
            )
        for k, v in extender._dict.items():
            setattr(self, k, getattr(self, k) + [v])


@dataclass
class KAddSpec(KSpecBase[KAddSpecExtender]):
    """
    Specification for adding knowledge entries.
    Adds timestamp.

    :ivar documents: List of documents to add.
    :ivar metadatas: List of metadata dictionaries.
    :ivar ids: List of document identifiers.
    """

    documents: Documents
    metadatas: Metadatas
    ids: IDs

    @classmethod
    @override
    def new(cls) -> Self:
        return cls([], [], [], [])


@dataclass
class KSearchSpec(KSpecBase[KSearchSpecExtender]):
    """
    Specification for knowledge search operations.

    :ivar instructions: List of query instructions.
    :ivar min_or_max_distances: List of distance thresholds.
    :ivar _n_results: Number of results to return.
    :ivar _instruct_task: Instruction task type.
    :ivar _rerank_task: Reranking task type.
    :ivar _with_reranker: Flag to enable reranker.
    :ivar _reranker_context: Flag to include context in reranking.
    :ivar _where: Metadata filtering conditions.
    :ivar _where_documents: Document content filtering conditions.
    :ivar _ids: Specific document IDs to search.
    """

    instructions: list[str]
    min_or_max_distances: list[float]
    # Unextendable.
    _n_results: int
    _instruct_task: str
    _rerank_task: str
    _with_reranker: bool
    _reranker_context: bool
    _where: Where | None
    _where_documents: WhereDocument | None
    _ids: IDs | None

    @classmethod
    @override
    def new(
        cls,
        n_results: int,
        instruct_task: str = '',
        rerank_task: str = '',
        with_reranker: bool = False,
        reranker_context: bool = False,
        where: Where | None = None,
        where_documents: WhereDocument | None = None,
        ids: IDs | None = None,
    ) -> Self:
        """
        Create new search specification.

        :param n_results: Number of results to return.
        :param instruct_task: Instruction task type.
        :param rerank_task: Reranking task type.
        :param with_reranker: Enable reranking.
        :param reranker_context: Include context in reranking.
        :param where: Metadata filtering conditions.
        :param where_documents: Document content filtering conditions.
        :param ids: Specific document IDs to search.
        :return: Initialized KSearchSpec instance.
        """
        return cls(
            [],
            [],
            [],
            n_results,
            instruct_task,
            rerank_task,
            with_reranker,
            reranker_context,
            where,
            where_documents,
            ids,
        )

    def _flush(self) -> None:
        """Clear extendable specification components."""
        self.handles = []
        self.instructions = []
        self.min_or_max_distances = []


@dataclass
class KReturnSpec(KSpecBase[NotExtendable]):
    """
    Specification for knowledge search results.

    :ivar instructions: Original query instructions.
    :ivar min_or_max_distances: Distance thresholds used.
    :ivar n_results_actual: Actual number of results returned.
    :ivar n_results: Requested number of results.
    :ivar instruct_task: Instruction task type used.
    :ivar with_reranker: Reranker usage flag.
    :ivar reranker_context: Context usage in reranking.
    :ivar where: Applied metadata filters.
    :ivar where_documents: Applied document content filters.
    :ivar ids: Applied ID filters.
    :ivar query_results: Tuple of query result objects.
    """

    instructions: list[str]
    min_or_max_distances: list[float]
    n_results_actual: list[int]
    n_results: int
    instruct_task: str
    with_reranker: bool
    reranker_context: bool
    where: Where | None
    where_documents: WhereDocument | None
    ids: IDs | None
    query_results: tuple[QueryResult, ...]

    @override
    @classmethod
    def new(cls) -> Self:
        """
        Not implemented factory method.

        :raises KnowledgeSpecError: Always raises since not implemented.
        """
        raise KnowledgeSpecError('Not implemented.') from NotImplementedError

    def _distance_cull(self) -> None:
        """Filter results based on distance thresholds."""
        qrs = self.query_results
        mds = self.min_or_max_distances
        operation = operator.ge if self.with_reranker else operator.le

        def get_k(
            m: float, qr: QueryResult, op: Callable[[float, float], bool]
        ) -> int:
            if (qrd := qr['distances']) is None:
                return 0

            ds = qrd[0]
            r = 0
            for d in ds:
                if not op(d, m):
                    return r
                r += 1
            else:
                return r

        self.query_results = tuple(
            QueryResult(
                distances=None
                if (di := qr['distances']) is None
                else [di[0][:k]],
                ids=[qr['ids'][0][:k]],
                embeddings=qr['embeddings'],
                documents=None
                if (do := qr['documents']) is None
                else [do[0][:k]],
                uris=None if (ur := qr['uris']) is None else [ur[0][:k]],
                included=qr['included'],
                data=qr['data'],
                metadatas=None
                if (me := qr['metadatas']) is None
                else [me[0][:k]],
            )
            for k, qr in (
                (
                    get_k(md, qr, operation),
                    qr,
                )
                for md, qr in zip(mds, qrs)
            )
        )

        self.n_results_actual = [
            len(qr['ids'][0]) for qr in self.query_results
        ] or [0]


@dataclass
class Knowledge(ABC, Mapping[T, U]):
    """
    Abstract knowledge base implementing dictionary-like interface.

    :ivar domain_name: Knowledge domain identifier
    :ivar db: Database provider instance
    """

    domain_name: str
    db: DatabaseProvider
    length: int = 0
    iter: Iterator[T] = iter(())

    @override
    def __len__(self) -> int:
        return self.length

    @override
    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError  # FIXME: Implement

    @abstractmethod
    def __perform_getitem__(self, spec: T) -> U:
        """Implementation of `__getitem__`"""

    @abstractmethod
    def __perform_setitem__(self, spec: T, data: KAddSpec) -> None:
        """Implementation of `__setitem__`"""

    @abstractmethod
    def simple_add(self, spec: KAddSpec) -> None:
        """Simple insertion."""

    @override
    def __getitem__(self, spec: T) -> U:
        """
        Retrieve knowledge using search specification.

        :param spec: Search specification.
        :return: Result specification.
        """
        ret = self.__perform_getitem__(spec)
        spec._flush()  # Flush the search spec each time.
        ret._distance_cull()  # Cull return spec based on `max_distances`.
        return ret

    # NOTE: Usecase:
    # Permanent collection X, temporary collection Y
    # This function merges query results from Y with X.
    # Y is cleared afterwards.
    def __setitem__(self, spec: T, data: KAddSpec) -> None:
        """
        Merge return data against a search specification.

        :param spec: Search specification
        :param data: Result data to merge
        """
        ret = self.__perform_setitem__(spec, data)
        spec._flush()  # Flush the search spec each time.
        return ret


@dataclass
class RAGKnowledge(Knowledge[KSearchSpec, KReturnSpec]):
    """
    Retrieval-Augmented Generation knowledge implementation.

    Provides concrete database operations for RAG systems.
    """

    def __post_init__(self) -> None:
        """Touch the domain."""
        self.db.touch(self.domain_name)

    @override
    def __perform_getitem__(self, spec: KSearchSpec) -> KReturnSpec:
        """
        Execute search operation using database provider.

        :param spec: Search specification.
        :return: Result specification with query results.
        :raises RuntimeError: If metadatas or distances are missing.
        """
        query_result = self.db.query(
            domain_name=self.domain_name,
            query_texts=spec.instructions,
            n_results=spec._n_results,
            instruct_task=spec._instruct_task,
            with_reranker=spec._with_reranker,  # Rerank before culling.
            reranker_context=spec._reranker_context,
            where=spec._where,
            where_document=spec._where_documents,
            ids=spec._ids,
        )
        split_qr = self._split_qr(query_result)

        return KReturnSpec(
            handles=spec.handles,
            instructions=spec.instructions,
            n_results_actual=[],
            n_results=spec._n_results,
            min_or_max_distances=spec.min_or_max_distances,
            instruct_task=spec._instruct_task,
            with_reranker=spec._with_reranker,
            reranker_context=spec._reranker_context,
            where=spec._where,
            where_documents=spec._where_documents,
            ids=spec._ids,
            query_results=split_qr,
        )

    @override
    def __perform_setitem__(self, spec: KSearchSpec, data: KAddSpec) -> None:
        """Placeholder for result merging."""  # FIXME: Implement.
        pass

    @override
    def simple_add(self, spec: KAddSpec) -> None:
        """
        Add documents to the knowledge base.

        :param spec: Add specification containing documents.
        """
        self.length += len(spec.ids)
        self.db.add(
            domain_name=self.domain_name,
            documents=spec.documents,
            metadatas=spec.metadatas,
            ids=spec.ids,
        )
        logger.debug(
            f'Added {len(spec.documents)} items to '
            f'{self.db}/{self.domain_name}.'
        )

    @staticmethod
    def _split_qr(qr: QueryResult) -> tuple[QueryResult, ...]:
        """
        Split combined query results into individual results.

        :param qr: Combined query results.
        :return: Tuple of individual query results.
        :raises RuntimeError: If metadatas or distances are None.
        """
        idsl = qr['ids']
        metadatasl = qr['metadatas']
        distancesl = qr['distances']

        documentsl = qr['documents'] or None
        urisl = qr['uris'] or None

        if metadatasl is None or distancesl is None:
            raise RuntimeError('Metadatas/distances should not be `None`.')

        if not all((metadatasl, distancesl)):  # Empty result
            return ()

        embeddings = qr['embeddings']
        included = qr['included']
        data = qr['data']

        return tuple(
            QueryResult(
                ids=[idsl[k]],
                embeddings=embeddings,
                documents=None if documentsl is None else [documentsl[k]],
                uris=None if urisl is None else [urisl[k]],
                included=included,
                data=data,
                metadatas=[metadatasl[k]],
                distances=[distancesl[k]],
            )
            for k in range(len(idsl))
        )
