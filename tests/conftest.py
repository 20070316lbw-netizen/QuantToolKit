import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def price_panel():
    # B 缺少部分日期且历史更短, A/B 的收益模式不同, 输入故意打乱。
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"]
    )
    a = pd.DataFrame(
        {"adj_close": [10.0, 20.0, 10.0, 30.0, 60.0]},
        index=pd.MultiIndex.from_product([dates, ["A"]], names=["date", "ticker"]),
    )
    b = pd.DataFrame(
        {"adj_close": [100.0, 100.0, 200.0]},
        index=pd.MultiIndex.from_product(
            [dates[[0, 2, 4]], ["B"]], names=["date", "ticker"]
        ),
    )
    panel = pd.concat([a, b]).sample(frac=1, random_state=42)
    panel["close"] = panel["adj_close"] * np.where(
        panel.index.get_level_values("ticker") == "A", 2.0, 3.0
    )
    return panel
