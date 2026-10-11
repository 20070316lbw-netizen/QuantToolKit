"""基于 (date, ticker) MultiIndex DataFrame 的波动率计算"""

from __future__ import annotations

import pandas as pd

from quanttoolkit.transfer import to_date_ticker_frame

