#!/usr/bin/env python3
from typing import Iterable
from util.ara_id import AraIdSupply
from util.bars import (
    Bar,
    EdgePart,
    Trigger,
    BarOp,
    IdClasses as BarsIdClasses,
    TriggerHook,
    TriggerResult,
    WindowPart,
)


# Initialize ID supply
ICl = BarsIdClasses
supply = AraIdSupply(ICl)


# Define hooks for triggers
@TriggerHook
def too_hot_hook(res: TriggerResult, bar: Bar):
    if Trigger.marked('TooHot', res):
        print(
            f'WARNING: Temperature is too hot! Current: {bar.cur}.',
            f'Cooling down to {bar.cur - 5.0}...',
        )
        return bar.new_decr(5.0, safe=True), {'action': 'cooled'}
    return BarOp(), {}


@TriggerHook
def too_cold_hook(res: TriggerResult, bar: Bar):
    if Trigger.marked('TooCold', res):
        print(
            f'WARNING: Temperature is too cold! Current: {bar.cur}.',
            f'Warming up to {bar.cur + 5.0}...',
        )
        return bar.new_incr(5.0, safe=True), {'action': 'warmed'}
    return BarOp(), {}


@TriggerHook
def safe_hook(res: TriggerResult, bar: Bar):
    if Trigger.marked('SafeRange', res):
        print(f'Temperature is safe. Current: {bar.cur}')
    return BarOp(), {}


# Create triggers
hot_trigger = Trigger(
    trigger_id=supply.next_id(ICl.TRIGGER),
    trigger_name='hot_trigger',
    trigger_list=[EdgePart('TooHot', 80.0, right=True)],
    trigger_hook=too_hot_hook,
)

cold_trigger = Trigger(
    trigger_id=supply.next_id(ICl.TRIGGER),
    trigger_name='cold_trigger',
    trigger_list=[EdgePart('TooCold', 20.0, left=True)],
    trigger_hook=too_cold_hook,
)

safe_trigger = Trigger(
    trigger_id=supply.next_id(ICl.TRIGGER),
    trigger_name='safe_trigger',
    trigger_list=[WindowPart('SafeRange', (20.0, 80.0), inside=True)],
    trigger_hook=safe_hook,
)


# Initialize the temperature bar
temperature_bar = Bar(
    bar_id=supply.next_id(ICl.BAR),
    cur=25.0,  # Initial temperature
    min=0.0,
    max=100.0,
    triggers=[hot_trigger, cold_trigger, safe_trigger],
    name='temperature',
    label='°C',
)


# Simulate temperature changes
def simulate_temperature_changes(bar: Bar, changes: Iterable[float]):
    print('Start bars example (temp=25.0°C).')
    for change in changes:
        print(f'\nAdjusting temperature by {change}°C...')
        if change > 0:
            op = bar.new_incr(change, safe=False)
        else:
            op = bar.new_decr(abs(change), safe=False)
        op()
        bar.run_triggers()


if __name__ == '__main__':
    temperature_changes = (10.0, 70.0, -20.0, -60.0, 50.0)
    simulate_temperature_changes(temperature_bar, temperature_changes)
