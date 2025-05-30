#!/usr/bin/env python3
from dataclasses import dataclass
from enum import Flag, auto
from functools import reduce, wraps
from operator import or_
from typing import Callable, Iterable, NamedTuple

from util.ara_id import AraId, IdClassBase


class TriggerHook:
    """Class wrapper for trigger hooks to ensure they match the expected signature."""

    def __init__(
        self, func: Callable[['TriggerResult', 'Bar'], tuple['BarOp', dict]]
    ):
        self.func = func
        wraps(func)(self)

    def __call__(
        self, res: 'TriggerResult', bar: 'Bar'
    ) -> tuple['BarOp', dict]:
        return self.func(res, bar)


X = 'UNIT'


class IdClasses(IdClassBase):
    """
    AraId classes for `util.bars`.

    :ivar TRIGGER: Class for triggers.
    :type TRIGGER: str
    :ivar BAR: Class for bars.
    :type BAR: str
    """

    TRIGGER = auto()
    BAR = auto()


@dataclass(frozen=True)
class TriggerPart:
    """
    A part of a `Trigger`.

    :param label: The label. Required.
    :type label: str
    """

    label: str


@dataclass(frozen=True)
class EdgePart(TriggerPart):
    """
    A trigger part for edge-based conditions.

    :param label: The label for the trigger part.
    :type label: str
    :param threshold: The threshold value for the edge condition.
    :type threshold: float
    :param left: If True, triggers when the value is below the threshold.
    :type left: bool | None
    :param right: If True, triggers when the value is above the threshold.
    :type right: bool | None
    :param active: If True, triggers on rising/falling past the threshold.
    :type active: bool
    """

    label: str
    threshold: float
    left: bool | None = None  # Before a threshold.
    right: bool | None = None  # Past a threshold.
    active: bool = False  # If true, triggers on rising/falling past threshold.


@dataclass(frozen=True)
class WindowPart(TriggerPart):
    """
    A trigger part for window-based conditions.

    :param window: A tuple of (min, max) defining the window bounds.
    :type window: tuple[float, float]
    :param inside: If True, triggers when the value is inside the window.
    :type inside: bool | None
    :param outside: If True, triggers when the value is outside the window.
    :type outside: bool | None
    :param active: If True, triggers on entering/exiting the window.
    :type active: bool
    :param label: The label for the trigger part.
    :type label: str
    """

    window: tuple[float, float]
    inside: bool | None = None  # Inside the window.
    outside: bool | None = None  # Outside the window.
    active: bool = False  # If true, triggers on entering/exiting the window.
    label: str


@dataclass(frozen=True)
class IntervalPart(TriggerPart):
    """
    A trigger part for interval-based conditions.

    :param interval: The interval value for the condition.
    :type interval: float
    :param exceed: If True, triggers when the change exceeds the interval.
    :type exceed: bool | None
    :param within: If True, triggers when the change stays within the interval.
    :type within: bool | None
    :param active: If True, the condition is active (no effect in this case).
    :type active: bool
    :param label: The label for the trigger part.
    :type label: str
    """

    interval: float
    exceed: bool | None = None  # Instantaneous change exceeds an interval.
    within: bool | None = None  # Instantaneous change stays within an interval.
    active: bool = True  # Does nothing.
    label: str


class TriggerResult(NamedTuple):
    """
    The result of a trigger check.

    :ivar match: The result of the trigger checks
    :type match: Flag
    :ivar ambient: All flags to check against. Created from triggerpart names.
    :type ambient: Flag
    """

    match: Flag
    ambient: Flag


@dataclass(frozen=True)
class Trigger:
    """
    A trigger that checks conditions on a bar's state and executes a hook when
    conditions are met.

    :ivar trigger_id: The unique identifier for the trigger.
    :vartype trigger_id: AraId
    :ivar trigger_name: The name of the trigger.
    :vartype trigger_name: str
    :ivar trigger_list: A list of trigger parts with the conditions to check.
    :vartype trigger_list: Iterable[TriggerPart]
    :ivar trigger_hook: A function that runs when trigger conditions are met.
    :vartype trigger_hook: TriggerHook
        The hook takes two arguments:
            - res (TriggerResult): The result of the trigger checks.
            - bar (Bar): The current bar state.
        It returns a tuple of:
            - BarOp: An operation to perform on the bar.
            - dict: Additional data or context.
    """

    trigger_id: AraId
    trigger_name: str
    trigger_list: Iterable[TriggerPart]
    trigger_hook: TriggerHook

    @property
    def ara_id(self) -> AraId:
        """Returns the id."""
        return self.trigger_id

    @staticmethod
    def _resolve_implicit(
        left: bool | None,
        right: bool | None,
        default: tuple[bool, bool] = (False, True),
    ) -> tuple[bool, bool]:
        """
        Resolves truth of two boolean or None options.

        :param left: The left parameter.
        :type left: bool
        :param right: The right parameter.
        :type right: bool
        :return: A tuple of resolved booleans.
        :rtype: tuple[bool, bool]
        """
        if left is not None and right is not None:
            return left, right
        elif left is not None and right is None:
            return left, not left
        elif left is None and right is not None:
            return not right, right
        else:
            return default

    @classmethod
    def _check_edge_trigger(
        cls, triggerpart: TriggerPart, prev_bar: 'Bar', cur_bar: 'Bar'
    ) -> bool:
        """
        Check if the bar's value meets the edge trigger conditions.

        :param triggerpart: The trigger part to check.
        :type triggerpart: TriggerPart
        :param prev_bar: The previous bar state.
        :type prev_bar: Bar
        :param cur_bar: The current bar state.
        :type cur_bar: Bar
        :return: True if the trigger conditions are met, False otherwise.
        :rtype: bool
        """
        if not isinstance(triggerpart, EdgePart):
            return False

        cur = cur_bar.cur
        threshold = triggerpart.threshold
        right, left = cls._resolve_implicit(triggerpart.right, triggerpart.left)
        if triggerpart.active:
            prev = prev_bar.cur

            return (right and prev <= threshold < cur) or (
                left and prev >= threshold > cur
            )
        else:
            return (right and cur > threshold) or (left and cur <= threshold)

    @classmethod
    def _check_window_trigger(
        cls, triggerpart: TriggerPart, prev_bar: 'Bar', cur_bar: 'Bar'
    ) -> bool:
        """
        Check if the bar's value meets the window trigger conditions.

        :param triggerpart: The trigger part to check.
        :type triggerpart: WindowTrigger
        :param prev_bar: The previous bar state.
        :type prev_bar: Bar
        :param cur_bar: The current bar state.
        :type cur_bar: Bar
        :return: True if the trigger conditions are met, False otherwise.
        :rtype: bool
        """
        if not isinstance(triggerpart, WindowPart):
            return False

        cur = cur_bar.cur
        window_min, window_max = triggerpart.window
        inside, outside = cls._resolve_implicit(
            triggerpart.inside, triggerpart.outside
        )
        if triggerpart.active:
            prev = prev_bar.cur

            return (
                inside
                and (prev < window_min or prev > window_max)
                and (window_min <= cur <= window_max)
            ) or (
                outside
                and (window_min <= prev <= window_max)
                and (cur < window_min or cur > window_max)
            )
        else:
            return (inside and window_min <= cur <= window_max) or (
                outside and (cur < window_min or cur > window_max)
            )

    @classmethod
    def _check_interval_trigger(
        cls, triggerpart: TriggerPart, prev_bar: 'Bar', cur_bar: 'Bar'
    ) -> bool:
        """
        Check if the bar's change meets the active interval trigger conditions.

        :param triggerpart: The trigger part to check.
        :type triggerpart: TriggerPart
        :param prev_bar: The previous bar state.
        :type prev_bar: Bar
        :param cur_bar: The current bar state.
        :type cur_bar: Bar
        :return: True if the trigger conditions are met, False otherwise.
        :rtype: bool
        """
        if not isinstance(triggerpart, IntervalPart):
            return False

        delta = abs(cur_bar.cur - prev_bar.cur)
        interval = triggerpart.interval
        within, exceed = cls._resolve_implicit(
            triggerpart.within, triggerpart.exceed
        )
        return (exceed and delta > interval) or (within and delta <= interval)

    def check(self, bar: 'Bar') -> TriggerResult:
        """
        Check all trigger parts comprising the trigger against the current and
        previous bar states.

        :param bar: The current bar state. This should not be created as a
            copy (see `Bar`).
        :type bar: Bar
        :return: A `TriggerResult` object.
        :rtype: TriggerResult
        """
        if bar._prev_bar is None:  # For the type checker :)
            raise ValueError(f'Recieved a bar with no history:\n{bar}')

        check_functions = (
            self._check_edge_trigger,
            self._check_window_trigger,
            self._check_interval_trigger,
        )

        try:
            flags: Flag = Flag(
                value=self.trigger_name,
                names=[(X, 0)] + [(t.label, auto()) for t in self.trigger_list],
            )
        except TypeError as e:
            raise TypeError(f'Do not use {X} as a `TriggerPart` label!') from e

        result = flags[X]  # pyright: ignore [reportIndexIssue]
        for trigger in self.trigger_list:
            if any(fn(trigger, bar._prev_bar, bar) for fn in check_functions):
                result |= flags[trigger.label]  # pyright: ignore [reportIndexIssue]

        return TriggerResult(result, flags)

    @staticmethod
    def marked(label: str, result: TriggerResult) -> bool:
        """
        Checks if a specific label is marked in the given `TriggerResult`.

        :param label: The label to check in the `TriggerResult`.
        :type label: str
        :param result: The `TriggerResult` object.
        :type result: TriggerResult
        :return: Whether or not the label is present in the match flags.
        :rtype: bool
        :raises KeyError: If the `label` does not exist in the ambient flags of
            the `TriggerResult`.
        """
        try:
            flag = result.ambient[label]  # pyright: ignore [reportIndexIssue]
        except KeyError as e:
            raise KeyError(
                f'Label {label} does not exist in {result.ambient}.'
            ) from e
        return flag in result.match

    def run_hook(self, bar: 'Bar') -> tuple['BarOp', dict]:
        """
        Runs the passed hook.

        :param bar:
        :type bar: Bar
        """
        return self.trigger_hook(self.check(bar), bar)


class BarOp:
    """Simple decorator for stateless bar operations."""

    def __init__(self, *funcs: Callable):
        self.funcs = funcs

    def __call__(self) -> tuple:
        return tuple(f() for f in self.funcs)

    def __or__(self, o: 'BarOp') -> 'BarOp':
        return BarOp(*self.funcs, *o.funcs)


@dataclass
class Bar:
    """
    A class representing a bar with a current value, min/max bounds, and
    triggers.

    :ivar bar_id: The unique identifier for the bar.
    :vartype bar_id: AraId
    :ivar cur: The current value of the bar.
    :vartype cur: float
    :ivar min: The minimum allowed value for the bar.
    :vartype min: float
    :ivar max: The maximum allowed value for the bar.
    :vartype max: float
    :ivar triggers: A single trigger or iterable of triggers associated with the
        bar. This will be converted to a list in `__post_init__`.
    :vartype triggers: Trigger | Iterable[Trigger]
    :ivar name: The name of the bar. If not provided, defaults to the string
        representation of `bar_id`.
    :vartype name: str | None
    :ivar label: An optional label for the bar.
    :vartype label: str | None
    :ivar _prev_bar: The previous state of the bar.
    :vartype _prev_bar: Bar | None
    :ivar _is_copy: A flag indicating whether the bar is a copy.
    :vartype _is_copy: bool
    """

    bar_id: AraId
    cur: float
    min: float
    max: float
    triggers: Trigger | Iterable[Trigger]
    name: str | None = None
    label: str | None = None
    _prev_bar: 'Bar | None' = None
    _is_copy: bool = False

    def __post_init__(self) -> None:
        """
        Initialize the bar's properties post-creation.

        - Sets `name` to `str(id)` if not provided.
        - Converts `triggers` to a list if it's a single `Trigger`.
        - Creates a copy of the bar for tracking states (`_prev_bar`, depth 1)
          if the bar is itself NOT a copy (`_is_copy` is false).
        """
        if self.name is None:
            self.name = str(self.bar_id)
        if not isinstance(self.triggers, list):
            if isinstance(self.triggers, Trigger):
                self.triggers = [self.triggers]
            else:
                self.triggers = list(self.triggers)
        if not self._is_copy:
            self._prev_bar = Bar._copy(self)

    def _copy(self) -> 'Bar':
        """
        Create a shallow copy of the bar.

        :return: A new `Bar` instance with the same properties.
        :rtype: Bar
        """
        return Bar(
            bar_id=self.bar_id,
            name=self.name,
            label=self.label,
            cur=self.cur,
            min=self.min,
            max=self.max,
            triggers=self.triggers,
            _is_copy=True,
        )

    @property
    def ara_id(self) -> AraId:
        """
        Get the bar's unique identifier.

        :return: The `AraId` of the bar.
        :rtype: AraId
        """
        return self.bar_id

    def new_set(self, a: float, safe: bool = True) -> BarOp:
        """
        Create an operation to set the bar's value.

        :param a: The new value.
        :type a: float
        :param safe: If `True`, ensures the value does not exceed `max`.
        :type safe: bool
        :return: A `BarOp` representing the increment operation.
        :rtype: BarOp
        """

        @BarOp
        def _set_cur() -> None:
            self.cur = a if not safe else max(min(a, self.max), self.min)

        return _set_cur

    def new_incr(self, a: float, safe: bool = True) -> BarOp:
        """
        Create an operation to increment the bar's current value.

        :param a: The amount to increment by.
        :type a: float
        :param safe: If `True`, ensures the value does not exceed `max`.
        :type safe: bool
        :return: A `BarOp` representing the increment operation.
        :rtype: BarOp
        """

        @BarOp
        def _set_cur() -> None:
            new = self.cur + a
            self.cur = new if not safe else min(new, self.max)

        return _set_cur

    def new_decr(self, a: float, safe: bool = True) -> BarOp:
        """
        Create an operation to decrement the bar's current value.

        :param a: The amount to decrement by.
        :type a: float
        :param safe: If `True`, ensures the value does not fall below `min`.
        :type safe: bool
        :return: A `BarOp` representing the decrement operation.
        :rtype: BarOp
        """

        @BarOp
        def _set_cur() -> None:
            new = self.cur - a
            self.cur = new if not safe else max(new, self.min)

        return _set_cur

    def new_set_max(self, a: float) -> BarOp:
        """
        Create an operation to set the bar's maximum value. This is limited by
        the bar's minimum value.

        :param a: The new value.
        :type a: float
        :return: A `BarOp` representing the increment operation.
        :rtype: BarOp
        """

        @BarOp
        def _set_cur() -> None:
            self.max = max(a, self.min)

        return _set_cur

    def new_incr_max(self, a: float) -> BarOp:
        """
        Create an operation to increment the bar's maximum value.

        :param a: The amount to increment by.
        :type a: float
        :return: A `BarOp` representing the increment operation.
        :rtype: BarOp
        """

        @BarOp
        def _set_max() -> None:
            self.max += a

        return _set_max

    def new_decr_max(self, a: float) -> BarOp:
        """
        Create an operation to decrement the bar's maximum value. This is
        limited by the bar's minimum value.

        :param a: The amount to decrement by.
        :type a: float
        :return: A `BarOp` representing the decrement operation.
        :rtype: BarOp
        """

        @BarOp
        def _set_max() -> None:
            self.max -= a
            self.max = max(self.min, self.max)

        return _set_max

    def new_set_min(self, a: float) -> BarOp:
        """
        Create an operation to set the bar's minimum value. This is limited by
        the bar's maximum value.

        :param a: The new value.
        :type a: float
        :return: A `BarOp` representing the increment operation.
        :rtype: BarOp
        """

        @BarOp
        def _set_cur() -> None:
            self.max = min(a, self.max)

        return _set_cur

    def new_incr_min(self, a: float) -> BarOp:
        """
        Create an operation to increment the bar's minimum value. This is
        limited by the bar's maximum value.

        :param a: The amount to increment by.
        :type a: float
        :return: A `BarOp` representing the increment operation.
        :rtype: BarOp
        """

        @BarOp
        def _set_min() -> None:
            self.min += a
            self.min = min(self.min, self.max)

        return _set_min

    def new_decr_min(self, a: float) -> BarOp:
        """
        Create an operation to decrement the bar's minimum value.

        :param a: The amount to decrement by.
        :type a: float
        :return: A `BarOp` representing the decrement operation.
        :rtype: BarOp
        """

        @BarOp
        def _set_min() -> None:
            self.min -= a

        return _set_min

    def run_operation(self, op: BarOp) -> None:
        """
        Runs a bar operation and updates `_prev_bar`.

        :param ops: The operation.
        :type ops: BarOp
        """
        prev_bar = Bar._copy(self)
        _ = op()
        self._prev_bar = prev_bar

    def run_triggers(self) -> dict:
        """
        Run all triggers associated with the bar, combine their operations, and
        merge their output dictionaries.

        :return: A merged dictionary of outputs from all triggers.
        :rtype: dict
        """
        if isinstance(self.triggers, Trigger):
            raise TypeError(
                'Internal variable self.triggers is not a list. '
                'This should never occur.'
            )

        ops, dicts = zip(*(trigger.run_hook(self) for trigger in self.triggers))

        if ops:
            self.run_operation(reduce(or_, ops))

        if dicts:
            return reduce(or_, dicts)
        else:
            return {}


@dataclass
class BarManager:
    """
    A class representing bars.

    :ivar bars: The bars.
    :type bars: list[Bar]
    """

    bars: list[Bar]

    def get_bar_by_id(self, ara_id: AraId) -> Bar | None:
        """
        Get a bar by its id.

        :param want: The id.
        :type want: AraId
        :return: A bar in bars, or nothing.
        :rtype: Bar | None
        """
        for bar in self.bars:
            if bar.bar_id == ara_id:
                return bar
        return None

    def remove_bar_by_id(self, ara_id: AraId) -> None:
        """
        Remove a bar from the list by its id.

        :param ara_id: The id.
        :type ara_id: AraId
        """
        self.bars = [bar for bar in self.bars if bar.bar_id != ara_id]
