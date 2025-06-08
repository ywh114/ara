#!/usr/bin/env python3
# TODO: Rewrite
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Generic, Self, TypeVar, override

from chromadb import Metadata, QueryResult
from llm.database import DatabaseProvider
from util.exceptions import KnowledgeSpecError
from util.logger import get_logger

T = TypeVar('T', bound='KSearchSpec')
U = TypeVar('U', bound='KReturnSpec')

V = TypeVar('V', bound='KSpecExtender')

logger = get_logger(__name__)


@dataclass
class KSpecExtender:
    handle: Any

    def __post_init__(self) -> None:
        self._dict = {
            f'{name}s': getattr(self, name)
            for x in fields(self)
            if not (name := x.name).startswith('_')
        }


@dataclass
class KSearchSpecExtender(KSpecExtender):
    instruction: str
    min_or_max_distance: float


@dataclass
class KAddSpecExtender(KSpecExtender):
    document: str
    metadata: Metadata
    id: str


@dataclass
class NotExtendable(KSpecExtender):
    pass


KSSExtender = KSearchSpecExtender
KASExtender = KAddSpecExtender


# TODO: Integrate timestamp.
@dataclass
class KSpecBase(ABC, Generic[V]):
    handles: list[Any]

    def __post_init__(self) -> None:
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
        self._timestamp: datetime = datetime.now()

    @classmethod
    @abstractmethod
    def new(cls, *args, **kwargs) -> Self: ...

    @classmethod
    def _get_x(cls) -> type[V]:
        """
        Extract extender type from generic parameter.

        :return: The extender class type.
        :rtype: type[T]
        :raises SpecError: If constraints not followed.
        """
        for base in getattr(cls, '__orig_bases__', ()):
            if hasattr(base, '__args__') and base.__args__:
                arg = base.__args__[0]
                if not isinstance(arg, TypeVar):
                    return arg

        raise KnowledgeSpecError('Do not use bare `KSpecBase`.')

    def get_handle(self, handle: Any) -> int:
        try:
            return self.handles.index(handle)
        except ValueError as e:
            raise ValueError(
                f'Handle {handle} not registered: {self.handles}.'
            ) from e

    def pop_handle(self, handle: Any) -> V:
        k = self.get_handle(handle)

        return self._get_x()(
            **{
                name[:-1]: getattr(self, name).pop(k)
                for fld in fields(self)
                if not (name := fld.name).startswith('_')
            }
        )

    def extend(self, extender: V) -> None:
        if not isinstance(extender, self._get_x()):
            raise KnowledgeSpecError(
                f'Wrong spec: expected {self._get_x()}, not {type(extender)}.'
            )
        for k, v in extender._dict.items():
            setattr(self, k, getattr(self, k) + [v])


@dataclass
class KAddSpec(KSpecBase[KAddSpecExtender]):
    documents: list[str]
    metadatas: list[Metadata]
    ids: list[str]

    @classmethod
    @override
    def new(cls) -> Self:
        return cls([], [], [], [])


# TODO: Rewrite
@dataclass
class KSearchSpec(KSpecBase[KSearchSpecExtender]):
    instructions: list[str]
    min_or_max_distances: list[float]
    # Unextendable.
    _n_results: int
    _instruct_task: str
    _rerank_task: str
    _with_reranker: bool
    _where: dict[str, str] | None
    _where_documents: dict[str, str] | None
    _ids: list[str] | None

    @classmethod
    @override
    def new(
        cls,
        n_results: int,
        instruct_task: str = '',
        rerank_task: str = '',
        with_reranker: bool = False,
        where: dict[str, str] | None = None,
        where_documents: dict[str, str] | None = None,
        ids: list[str] | None = None,
    ) -> Self:
        return cls(
            [],
            [],
            [],
            n_results,
            instruct_task,
            rerank_task,
            with_reranker,
            where,
            where_documents,
            ids,
        )

    def _flush(self) -> None:
        self.handles = []
        self.instructions = []
        self.min_or_max_distances = []


@dataclass
class KReturnSpec(KSpecBase[NotExtendable]):
    instructions: list[str]
    min_or_max_distances: list[float]
    n_results_actual: list[int]
    n_results: int
    instruct_task: str
    with_reranker: bool
    where: dict[str, str] | None
    where_documents: dict[str, str] | None
    ids: list[str] | None
    query_results: tuple[QueryResult, ...]

    @override
    @classmethod
    def new(cls) -> Self:
        raise KnowledgeSpecError('Not implemented.') from NotImplementedError

    def _distance_cull(self) -> None:
        qrs = self.query_results
        ds = self.min_or_max_distances
        operation = operator.ge if self.with_reranker else operator.le
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
                    len(
                        tuple(
                            r
                            for r in (
                                qr['distances'][0]
                                if qr['distances'] is not None
                                else [d + 1]
                            )
                            if operation(r, d)
                        )
                    ),
                    qr,
                )
                for d, qr in zip(ds, qrs)
            )
        )

        self.n_results_actual = [len(qr['ids'][0]) for qr in self.query_results]


@dataclass
class Knowledge(ABC, dict[T, U]):
    domain_name: str
    db: DatabaseProvider

    @abstractmethod
    def __perform_getitem__(self, spec: T) -> U:
        """Implementation of `__getitem__`"""

    @abstractmethod
    def __perform_setitem__(self, spec: T, data: U) -> None:
        """Implementation of `__setitem__`"""

    @abstractmethod
    def simple_add(self, spec: KAddSpec) -> None:
        """Simple insertion."""

    @override
    def __getitem__(self, spec: T) -> U:
        """Get a spec."""
        ret = self.__perform_getitem__(spec)
        spec._flush()  # Flush the search spec each time.
        ret._distance_cull()  # Cull return spec based on `max_distances`.
        return ret

    # NOTE: Usecase:
    # Permanent collection X, temporary collection Y
    # This function merges query results from Y with X.
    # Y is cleared afterwards.
    @override
    def __setitem__(self, spec: T, data: U) -> None:
        """Merge return data against a spec."""
        ret = self.__perform_setitem__(spec, data)
        spec._flush()  # Flush the search spec each time.
        return ret


@dataclass
class RAGKnowledge(Knowledge[KSearchSpec, KReturnSpec]):
    @override
    def __perform_getitem__(self, spec: KSearchSpec) -> KReturnSpec:
        query_result = self.db.query(
            domain_name=self.domain_name,
            query_texts=spec.instructions,
            n_results=spec._n_results,
            instruct_task=spec._instruct_task,
            with_reranker=spec._with_reranker,  # Rerank before culling.
            where=spec._where,
            where_document=spec._where_documents,
            ids=spec._ids,
        )
        split_qr = self._split_qr(query_result)

        return KReturnSpec(
            handles=spec.handles,  # Pass through.
            instructions=spec.instructions,  # Pass through.
            n_results_actual=[],
            n_results=spec._n_results,  # Pass through for now.
            min_or_max_distances=spec.min_or_max_distances,  # Pass through.
            instruct_task=spec._instruct_task,  # Pass through.
            with_reranker=spec._with_reranker,  # Pass through
            where=spec._where,
            where_documents=spec._where_documents,
            ids=spec._ids,
            query_results=split_qr,
        )

    @override
    def __perform_setitem__(self, spec: KSearchSpec, data) -> None:
        pass

    @override
    def simple_add(self, spec: KAddSpec) -> None:
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
        idsl = qr['ids']
        metadatasl = qr['metadatas']
        distancesl = qr['distances']

        documentsl = qr['documents'] or None
        urisl = qr['uris'] or None

        if metadatasl is None or distancesl is None:
            raise RuntimeError('Metadatas/distances should not be `None`.')

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
