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
from typing import Iterable
from uuid import uuid4

from utils.bars import (
    Bar,
    BarOp,
    EdgePart,
    Trigger,
    TriggerHook,
    TriggerResult,
    WindowPart,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# Define hooks for triggers
@TriggerHook
def too_hot_hook(res: TriggerResult, bar: Bar) -> TriggerHook.RType:
    if Trigger.marked('TooHot', res):
        return bar.new_decr(5.0, safe=True), {'action': 'cooled'}


@TriggerHook
def too_cold_hook(res: TriggerResult, bar: Bar) -> TriggerHook.RType:
    if Trigger.marked('TooCold', res):
        return bar.new_incr(5.0, safe=True), {'action': 'warmed'}


@TriggerHook
def alert_hook(res: TriggerResult, _: Bar) -> TriggerHook.RType:
    if Trigger.marked('SafeRange', res):
        return BarOp(), {'alert': 'all is well'}
    elif Trigger.marked('TooHot', res):
        return BarOp(), {'alert': 'call the fire department'}
    elif Trigger.marked('TooCold', res):
        return BarOp(), {'alert': 'call the ice department'}
    else:
        return BarOp(), {'alert': 'alien invasion'}


# Create triggers
hot_trigger = Trigger(
    id=uuid4(),
    trigger_name='hot_trigger',
    trigger_list=[hotedge := EdgePart('TooHot', 80.0, right=True)],
    trigger_hook=too_hot_hook,
)

cold_trigger = Trigger(
    id=uuid4(),
    trigger_name='cold_trigger',
    trigger_list=[coldedge := EdgePart('TooCold', 20.0, left=True)],
    trigger_hook=too_cold_hook,
)

alert_trigger = Trigger(
    id=uuid4(),
    trigger_name='safe_trigger',
    trigger_list=[
        WindowPart('SafeRange', (20.0, 80.0), outside=False),
        hotedge,
        coldedge,
    ],
    trigger_hook=alert_hook,
)


# Initialize the temperature bar
temperature_bar = Bar(
    id=uuid4(),
    cur=25.0,  # Initial temperature
    min=0.0,
    max=100.0,
    triggers=[hot_trigger, cold_trigger, alert_trigger],
    name='temperature',
    label='°C',
)


# Simulate temperature changes
def simulate_temperature_changes(bar: Bar, changes: Iterable[float]):
    logger.info('Start bars example (temp=25.0°C).')
    for change in changes:
        logger.info(f'Adjusting temperature by {change}°C...')
        if change > 0:
            op = bar.new_incr(change, safe=False)
        else:
            op = bar.new_decr(abs(change), safe=False)
        op()
        logger.info(f'Returned: {bar.run_triggers()}')
        logger.info(f'Current: {bar.cur}')


if __name__ == '__main__':
    temperature_changes = (10.0, 70.0, -20.0, -60.0, 50.0)
    simulate_temperature_changes(temperature_bar, temperature_changes)
