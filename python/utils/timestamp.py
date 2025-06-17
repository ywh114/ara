#!/usr/bin/env python3
from datetime import datetime


def suffixed_day(day: int) -> str:
    if 4 <= day <= 20 or 24 <= day <= 30:
        return f'{day}th'
    else:
        suffixes = ('st', 'nd', 'rd')
        return f'{day}{suffixes[min(day % 10 - 1, 2)]}'


today = datetime.today()

year = today.year
month = today.month
day = today.day

day_of_week_name = today.strftime('%A')
month_name = today.strftime('%B')
day_name = suffixed_day(day)
