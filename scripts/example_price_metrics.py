"""离线演示收益率、波动率和共同日历调仓, 不需要行情下载或数据库。"""

import pandas as pd

from quanttoolkit.rebalance import rebalance_dates_belong
from quanttoolkit.returns import future_returns, historical_return, log_returns
from quanttoolkit.transfer import to_date_ticker_frame
from quanttoolkit.volatility import forward_volatility, history_vol


def main() -> None:
    prices = to_date_ticker_frame(
        df=pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-02", periods=6),
                "ticker": ["A"] * 6,
                "adj_close": [100.0, 110.0, 99.0, 118.8, 118.8, 130.68],
            }
        )
    )
    metrics = pd.concat(
        [
            historical_return(df=prices, n_periods=2),
            future_returns(df=prices, n_periods=2, gap=1),
            log_returns(df=prices),
            history_vol(df=prices, window=2, price_col="adj_close"),
            forward_volatility(df=prices, window=2, gap=1, price_col="adj_close"),
        ],
        axis=1,
    )
    print(metrics.to_string())
    print(rebalance_dates_belong(prices.index.get_level_values("date"), freq=2))


if __name__ == "__main__":
    main()
