#!/usr/bin/env python3
from datetime import datetime
from datetime import UTC


class Timestamp:
    @staticmethod
    def suffix_day(day: int) -> str:
        if 4 <= day <= 20 or 24 <= day <= 30:
            return f'{day}th'
        else:
            suffixes = ('st', 'nd', 'rd')
            return f'{day}{suffixes[min(day % 10 - 1, 2)]}'

    @property
    def now(self) -> datetime:
        return datetime.now(UTC)

    def to_timestamp(self, dt: datetime) -> int:
        return int(dt.timestamp() * 1e6)

    @property
    def timestamp(self) -> int:
        return self.to_timestamp(self.now)

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
