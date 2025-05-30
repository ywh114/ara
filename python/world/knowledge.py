#!/usr/bin/env python3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Self, TypeVar, override
from itertools import chain

from chromadb import QueryResult
from llm.database import Database
from util.logger import get_logger

T = TypeVar('T', bound='KSearchSpec')
U = TypeVar('U', bound='KReturnSpec')

logger = get_logger(__name__)


@dataclass
class KSearchSpecExtender:
    handle: Any
    instruction: str
    n_results: int
    max_distance: float


KSSExtender = KSearchSpecExtender


@dataclass
class SpecBase:
    handles: list[Any]
    instructions: list[str]
    n_resultsl: list[int]
    max_distances: list[float]


@dataclass
class KSearchSpec(SpecBase):
    @classmethod
    def new(cls) -> Self:
        return cls([], [], [], [])

    def extend(self, extender: KSearchSpecExtender) -> None:
        self.handles += [extender.handle]
        self.instructions += [extender.instruction]
        self.n_resultsl += [extender.n_results]
        self.max_distances += [extender.max_distance]

    def _expand(self) -> Iterator[tuple[Any, str, int]]:
        return zip(self.handles, self.instructions, self.n_resultsl)

    def _flush(self) -> None:
        self.handles = []
        self.instructions = []
        self.n_resultsl = []
        self.max_distances = []


@dataclass
class KReturnSpec(SpecBase):
    query_results: tuple[QueryResult, ...]

    def get_handle(self, handle: Any) -> int:
        try:
            return self.handles.index(handle)
        except ValueError as e:
            raise ValueError(
                f'Handle {handle} not registered: {self.handles}.'
            ) from e

    def _distance_cull(self) -> None:
        qrs = self.query_results
        ds = self.max_distances
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
                            if r <= d
                        )
                    ),
                    qr,
                )
                for d, qr in zip(ds, qrs)
            )
        )

        self.n_resultsl = [len(qr['ids'][0]) for qr in self.query_results]


@dataclass
class Knowledge(ABC, dict[T, U]):
    domain_name: str
    db: Database

    @abstractmethod
    def __perform_getitem__(self, spec: T) -> U: ...

    @abstractmethod
    def __perform_setitem__(self, spec: T, data) -> None: ...

    @override
    def __getitem__(self, spec: T) -> U:
        ret = self.__perform_getitem__(spec)
        spec._flush()  # Flush the search spec each time.
        ret._distance_cull()  # Cull return spec based on `max_distances`.
        return ret

    @override
    def __setitem__(self, spec: T, data) -> None:
        ret = self.__perform_setitem__(spec, data)
        spec._flush()  # Flush the search spec each time.
        return ret


@dataclass
class RAGKnowledge(Knowledge[KSearchSpec, KReturnSpec]):
    @override
    def __perform_getitem__(self, spec: KSearchSpec) -> KReturnSpec:
        def _permute(a: Iterable, b: Iterable, t: tuple) -> tuple:
            index_map = {str(val): idx for idx, val in enumerate(a)}
            permutation_indices = (index_map[str(val)] for val in b)
            return tuple(t[i] for i in permutation_indices)

        grouped: dict[int, list[str]] = {}
        for handle, instruction, n in spec._expand():
            grouped.setdefault(n, [])
            grouped.setdefault(-n, [])
            grouped[n] += [instruction]
            grouped[-n] += [handle]

        # Search in batches based on n_results
        qrs = tuple(
            self.db.query(self.domain_name, grouped[k], k)
            for k in grouped
            if k > 0
        )

        logger.debug(
            f'Performed search on {self.db}/{self.domain_name} in '
            f'{len(grouped) // 2} batch(es).'
        )

        split_qrs = [self._split_qr(qr) for qr in qrs]

        handles = spec.handles
        permuted_handles = chain.from_iterable(
            grouped[k] for k in grouped if k < 0
        )

        return KReturnSpec(
            handles=spec.handles,  # Pass through
            instructions=spec.instructions,  # Pass through
            n_resultsl=spec.n_resultsl,  # Pass through
            max_distances=spec.max_distances,  # Pass through
            query_results=_permute(
                handles, permuted_handles, tuple(chain(*split_qrs))
            ),
        )

    @override
    def __perform_setitem__(self, spec: KSearchSpec, data) -> None:
        pass

    @staticmethod
    def _split_qr(qr: QueryResult) -> tuple[QueryResult, ...]:
        idsl = qr['ids']
        metadatasl = qr['metadatas']
        distancesl = qr['distances']

        documentsl = qr['documents'] or None
        urisl = qr['uris'] or None

        if metadatasl is None or distancesl is None:
            raise RuntimeError('Metadatas/distances should not be None')

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
