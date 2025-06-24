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
from utils.logger import get_logger
from world.importance import Importance, ImportanceType, imauto

logger = get_logger(__name__)

isplit = 1


# Define a custom Importance class for task priorities
class CharacterImportance(Importance, isplit=isplit):
    ANONYMOUS = imauto()
    IMPORTANT = imauto()
    REQUIRED = imauto()
    EIGEN = imauto()


# Simulate task prioritization
def demo_importance():
    logger.info(f'Start importance example (isplit={isplit})')

    # Create task priorities
    low_priority = CharacterImportance.ANONYMOUS
    medium_priority = CharacterImportance.IMPORTANT
    high_priority = CharacterImportance.REQUIRED
    level1_priority = CharacterImportance.ILevel(1.0)
    level2_priority = CharacterImportance.ILevel(2.0)

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
        f'\nisinstance({CharacterImportance},\n\t   {ImportanceType}): '
        f'{isinstance(CharacterImportance, ImportanceType)}'
    )
    logger.info(
        f'isinstance({low_priority},\n\t   {CharacterImportance}): '
        f'{isinstance(low_priority, Importance)}',
    )
    logger.info(
        f'isinstance({level1_priority},\n\t   {CharacterImportance}): '
        f'{isinstance(level1_priority, Importance)}'
    )


if __name__ == '__main__':
    demo_importance()
