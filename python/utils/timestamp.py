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
# XXX: %GREP_FLAG%
from datetime import UTC, datetime


# TODO: Implement.
class Timestamp:
    def __init__(self) -> None:
        self.time = datetime.now(UTC)

    @staticmethod
    def suffix_day(day: int) -> str:
        if 4 <= day <= 20 or 24 <= day <= 30:
            return f'{day}th'
        else:
            suffixes = ('st', 'nd', 'rd')
            return f'{day}{suffixes[min(day % 10 - 1, 2)]}'

    @property
    def real_now(self) -> datetime:
        return datetime.now(UTC)

    @property
    def real_timestamp(self) -> int:
        return self.to_timestamp(self.real_now)

    @property
    def now(self) -> datetime:
        return self.time

    def to_timestamp(self, dt: datetime) -> int:
        return int(dt.timestamp() * 1e6)

    @property
    def timestamp(self) -> int:
        return self.to_timestamp(self.now)

    @property
    def hour(self) -> int:
        return self.now.hour

    @property
    def minute(self) -> int:
        return self.now.minute

    @property
    def second(self) -> int:
        return self.now.second

    @property
    def today(self) -> datetime:
        return datetime.today()

    @property
    def year(self) -> int:
        return self.today.year

    @property
    def month(self) -> int:
        return self.today.month

    @property
    def day(self) -> int:
        return self.today.day

    @property
    def day_of_week_name(self) -> str:
        return self.today.strftime('%A')

    @property
    def month_name(self) -> str:
        return self.today.strftime('%B')

    @property
    def day_name(self) -> str:
        return self.suffix_day(self.day)


timestamp = Timestamp()
