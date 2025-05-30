#!/usr/bin/env python3
from util.importance import Importance, ImportanceType, imauto

isplit = 1


# Define a custom Importance class for task priorities
class TaskPriority(Importance, isplit=isplit):
    Low = imauto()
    Medium = imauto()
    High = imauto()


# Simulate task prioritization
def simulate_task_prioritization():
    print(f'Start importance example (isplit={isplit}).\n')

    # Create task priorities
    low_priority = TaskPriority.Low
    medium_priority = TaskPriority.Medium
    high_priority = TaskPriority.High
    level1_priority = TaskPriority.ILevel(1.0)
    level2_priority = TaskPriority.ILevel(2.0)

    # Print priorities
    print(f'Low Priority: {low_priority}')
    print(f'Medium Priority: {medium_priority}')
    print(f'High Priority: {high_priority}')
    print(f'Lvl1 Priority: {level1_priority}')
    print(f'Lvl2 Priority: {level2_priority}')

    # Compare priorities
    print('\nComparison Results:')
    print(f'Low < Medium: {low_priority < medium_priority}')
    print(f'Medium < High: {medium_priority < high_priority}')
    print(f'High < Low: {high_priority < low_priority}')
    print(f'Medium == Medium: {medium_priority == medium_priority}')

    # Dynamic priority comparisons
    print('\nDynamic Priorities:')
    print(f'Lvl1 > Low: {level1_priority > low_priority}')
    print(f'Lvl1 > Medium: {level1_priority > medium_priority}')
    print(f'Lvl1 < Lvl2: {level1_priority < level2_priority}')

    # Type checking
    print(
        f'\nisinstance({TaskPriority},\n\t   {ImportanceType}): '
        f'{isinstance(TaskPriority, ImportanceType)}'
    )
    print(
        f'isinstance({low_priority},\n\t   {TaskPriority}): '
        f'{isinstance(low_priority, Importance)}',
    )
    print(
        f'isinstance({level1_priority},\n\t   {TaskPriority}): '
        f'{isinstance(level1_priority, Importance)}'
    )


if __name__ == '__main__':
    simulate_task_prioritization()
