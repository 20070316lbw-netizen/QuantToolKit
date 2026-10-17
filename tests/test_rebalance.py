import numpy as np
import pandas as pd
import pytest

from quanttoolkit.portfolio.rebalance import rebalance_dates, rebalance_dates_belong


@pytest.mark.parametrize(
    "freq, positions, belongs",
    [
        (1, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
        (2, [0, 2, 4], [0, 0, 2, 2, 4]),
        (10, [0], [0, 0, 0, 0, 0]),
    ],
)
def test_common_calendar_sorted_unique_and_interval_boundaries(
    freq, positions, belongs
):
    calendar = (
        pd.DatetimeIndex(
            ["2024-01-02", "2024-01-05", "2024-01-08", "2024-01-11", "2024-01-12"],
            name="date",
        )
        .tz_localize("Asia/Shanghai")
        .as_unit("ms")
    )
    unsorted = calendar[[4, 1, 2, 0, 1, 3]]
    pd.testing.assert_index_equal(
        rebalance_dates(unsorted, freq=np.int64(freq)), calendar[positions]
    )
    pd.testing.assert_series_equal(
        rebalance_dates_belong(unsorted, freq=freq),
        pd.Series(calendar[belongs], index=calendar, name="rebalance_date"),
    )


def test_empty_calendar_retains_timezone_precision_and_name():
    dates = pd.DatetimeIndex([], dtype="datetime64[ms, Asia/Shanghai]", name="date")
    pd.testing.assert_index_equal(rebalance_dates(dates, freq=2), dates)
    pd.testing.assert_series_equal(
        rebalance_dates_belong(dates, freq=2),
        pd.Series(index=dates, dtype=dates.dtype, name="rebalance_date"),
    )
