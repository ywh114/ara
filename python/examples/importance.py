#!/usr/bin/env python3
from world.importance import Importance, ImportanceType, imauto
from util.logger import get_logger


logger = get_logger(__name__)

isplit = 1


# Define a custom Importance class for task priorities
class TaskPriority(Importance, isplit=isplit):
    LOW = imauto()
    MEDIUM = imauto()
    HIGH = imauto()


# Simulate task prioritization
def simulate_task_prioritization():
    logger.info(f'Start importance example (isplit={isplit})')

    # Create task priorities
    low_priority = TaskPriority.LOW
    medium_priority = TaskPriority.MEDIUM
    high_priority = TaskPriority.HIGH
    level1_priority = TaskPriority.ILevel(1.0)
    level2_priority = TaskPriority.ILevel(2.0)

    # logger.info priorities
    logger.info(f'Low Priority: {low_priority}')
    logger.info(f'Medium Priority: {medium_priority}')
    logger.info(f'High Priority: {high_priority}')
    logger.info(f'Lvl1 Priority: {level1_priority}')
    logger.info(f'Lvl2 Priority: {level2_priority}')

    # Compare priorities
    logger.info('Comparison Results:')
    logger.info(f'Low < Medium: {low_priority < medium_priority}')
    logger.info(f'Medium < High: {medium_priority < high_priority}')
    logger.info(f'High < Low: {high_priority < low_priority}')
    logger.info(f'Medium == Medium: {medium_priority == medium_priority}')

    # Dynamic priority comparisons
    logger.info('Dynamic Priorities:')
    logger.info(f'Lvl1 > Low: {level1_priority > low_priority}')
    logger.info(f'Lvl1 > Medium: {level1_priority > medium_priority}')
    logger.info(f'Lvl1 < Lvl2: {level1_priority < level2_priority}')

    # Type checking
    logger.info(
        f'\nisinstance({TaskPriority},\n\t   {ImportanceType}): '
        f'{isinstance(TaskPriority, ImportanceType)}'
    )
    logger.info(
        f'isinstance({low_priority},\n\t   {TaskPriority}): '
        f'{isinstance(low_priority, Importance)}',
    )
    logger.info(
        f'isinstance({level1_priority},\n\t   {TaskPriority}): '
        f'{isinstance(level1_priority, Importance)}'
    )


if __name__ == '__main__':
    simulate_task_prioritization()
