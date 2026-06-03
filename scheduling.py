from datetime import datetime, time
from typing import Optional

import pandas as pd
import pandas_market_calendars as mcal
import pytz


CENTRAL_TZ = pytz.timezone("America/Chicago")
NYSE_CALENDAR_NAME = "NYSE"
RUN_CUTOFF_CT = time(17, 0)


def to_central(dt: Optional[datetime] = None) -> datetime:
    if dt is None:
        return datetime.now(CENTRAL_TZ)
    if dt.tzinfo is None:
        return CENTRAL_TZ.localize(dt)
    return dt.astimezone(CENTRAL_TZ)


def is_after_cutoff(dt: Optional[datetime] = None, cutoff: time = RUN_CUTOFF_CT) -> bool:
    dt_ct = to_central(dt)
    return dt_ct.time() >= cutoff


def is_last_nyse_trading_day(dt: Optional[datetime] = None) -> bool:
    dt_ct = to_central(dt)
    today = pd.Timestamp(dt_ct.date())
    month_start = today.replace(day=1)
    month_end = today + pd.offsets.MonthEnd(0)

    calendar = mcal.get_calendar(NYSE_CALENDAR_NAME)
    schedule = calendar.schedule(start_date=month_start.date(), end_date=month_end.date())
    if schedule.empty:
        return False

    trading_dates = [idx.date() for idx in schedule.index]
    return dt_ct.date() == trading_dates[-1]


def should_run_now(dt: Optional[datetime] = None) -> bool:
    dt_ct = to_central(dt)
    return is_after_cutoff(dt_ct) and is_last_nyse_trading_day(dt_ct)
