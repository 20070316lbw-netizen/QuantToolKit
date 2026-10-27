# 作为依赖使用: 推荐调用顺序

忘记先调谁, 后调谁时, 先看这一页。这里用一份行情把因子, 分数和仓位接起来, 再加上模型分支。
安装方式见 [README](../README.md), 参数细节见各专题文档。窗口和模型参数只是为了演示调用顺序。

## 一眼速查

```text
数据入口 (三选一)
  Yahoo: get_prices(...)                           → prices
  数据库: get_duckdb(path=自己的路径) → SQL → .df()
          → to_date_ticker_frame(df=raw)            → prices
  自有长表: to_date_ticker_frame(df=raw)            → prices

直接使用因子
  prices → base_*factor → pd.concat(axis=1)         → factors
         → [neutralize_by_industry]                → 行业中性因子
         → zscore_by_date → combine_scores         → score
         → rebalance_dates → 筛选调仓日分数
         → simple_quantile_long_short              → weights

模型打分 (替换上面的 score 来源)
  factors + future_returns → assemble_samples      → samples
  完整行情日历 → split_dates(embargo=gap+n_periods) → 日期分段
  samples 按日期分段 → 训练 → 验证选型 → 测试       → pred
  pred → 筛选调仓日 → simple_quantile_long_short    → weights

事后评估 (从分数处单独分支)
  score 或 pred + future_returns → rank_ic → ic_summary
```

方括号表示可选步骤。单个因子可以直接作为 `score`, 不必标准化再合成。
模型特征可以直接用原始因子列, 不要求先调用 `combine_scores`。

| 现在想做什么 | 先调用 | 再调用 | 拿到什么 |
| --- | --- | --- | --- |
| 只算收益或波动率 | 准备 `prices` | `historical_return` / `history_vol` | 历史特征 Series |
| 看因子能否预测未来 | 因子函数 + `future_returns` | `ic` 或 `rank_ic` → `ic_summary` | 每日相关及汇总 |
| 多因子直接打分 | `pd.concat` | 可选行业中性 → `zscore_by_date` → `combine_scores` | `score` Series |
| 构造调仓目标仓位 | `rebalance_dates` → 筛选分数 | `simple_quantile_long_short` | 调仓日 `weight` Series |
| 按波动率分层选股 | 同时准备分数和 `history_vol` | 同步筛选调仓日 → `vol_quantile_long_short` | 分层多空仓位 |
| 用模型代替手工合成 | `assemble_samples` + `split_dates` | 训练 → 预测 → 恢复样本索引 | `pred` Series |
| 建一个买入持有基准 | 截取同一评估区间的 `prices` | `equal_weight_hold` | 基准单期收益 Series |

## 接起来时先看这几件事

- **计算和入库分别用什么表**: 计算输入通常是按 `[date, ticker]` 排序的 MultiIndex。`get_prices` 已经返回这个格式, 无需再转换; 数据库 `.df()` 的结果要转换。入库时反过来用 `prices.reset_index()`。
- **先算因子, 再截评估区间**: 先在包含预热历史的行情上算因子, 再截取需要的分数。先截价格可能让评估开头的窗口不够长。
- **特征和标签各放什么**: `future_returns` 和 `forward_volatility` 用于标签或事后评估, 不放进当日特征。包含当天收盘价的因子应在收盘数据可用后生成。
- **窗口按交易记录计数**: 收益与因子窗口按每只证券自己的记录数; 调仓和日期切分按共同日历。需要补齐时显式调用 `fill_price_gaps`, 它会沿用前价并将补齐日 volume 设为 0, 上市前和末条记录之后不补。
- **拿到仓位后还要做什么**: `weights` 只是调仓日的目标权重。后面还要接上实际成交时点, 持仓延续, 价格变化后的权重漂移, 费用和净值计算, 这些由使用项目的代码处理。目前本包没有统一的策略回测或净值接口。
- **缺失值按用途处理**: 因子预热和尾部未来标签的 NaN 是正常结果。`assemble_samples` 要求所有特征和标签完整; `combine_scores` 则允许只用该行不缺失的因子合成。

## 离线串起来: 数据 → 因子 → 分数 → 仓位

本节到“替换数据入口”之前的 Python 代码块按顺序执行, 无需网络或本地数据库。
换到自己的项目时, 保留调用顺序, 把第一步的演示行情换成真实数据即可。

### 1. 准备标准行情面板

```python
import numpy as np
import pandas as pd

from quanttoolkit.panel.transfer import to_date_ticker_frame

dates = pd.bdate_range("2024-01-02", periods=100, name="date")
tickers = ["A", "B", "C", "D", "E", "F"]
index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
rng = np.random.default_rng(7)
daily = rng.normal(0.0005, 0.015, size=(len(dates), len(tickers)))
raw = index.to_frame(index=False)
raw["adj_close"] = (100 * np.exp(daily.cumsum(axis=0))).ravel()
prices = to_date_ticker_frame(df=raw.copy())
calendar = prices.index.get_level_values("date").unique().sort_values()

assert prices.index.names == ["date", "ticker"]
assert prices.index.is_monotonic_increasing
```

### 2. 因子 → 标准化 → 合成分数

```python
from quanttoolkit.factors.combine import combine_scores
from quanttoolkit.factors.logvol_base import base_logvolfactor
from quanttoolkit.factors.mom_base import base_momfactor
from quanttoolkit.factors.zscore import zscore_by_date

factors = pd.concat(
    [base_momfactor(prices, window=21), base_logvolfactor(prices, window=21)],
    axis=1,
)
# 如需行业中性, 在这里把 factors 换成 neutralize_by_industry(factors, industry)
# industry 是以 ticker 为索引的行业标签 Series
z = zscore_by_date(factors)
score = combine_scores(z, weights={"mom_21": 1.0, "logvol_21": -1.0})

assert score.name == "score"
assert score.index.equals(prices.index)
assert score.loc[dates[:21]].isna().all()
```

权重 `-1.0` 表示低波动方向, 是演示选择。先让各因子的高低含义和尺度一致, 再合成分数,
具体规则见 [因子文档](factors.md)。

### 3. 分数 → 调仓日目标仓位

```python
from quanttoolkit.portfolio.rebalance import rebalance_dates
from quanttoolkit.portfolio.sizing import simple_quantile_long_short

# 先去掉因子预热区间, 再从可用分数的首日开始每 5 个交易日调仓
usable = score.dropna()
active_calendar = calendar[calendar >= usable.index.get_level_values("date").min()]
reb_dates = rebalance_dates(active_calendar, freq=5)
reb_score = usable.loc[usable.index.get_level_values("date").isin(reb_dates)]
weights = simple_quantile_long_short(score=reb_score, n_quantiles=3)

assert not weights.empty
assert weights.name == "weight"
assert weights.index.get_level_values("date").isin(reb_dates).all()
assert np.allclose(weights[weights > 0].groupby(level="date").sum(), 1.0)
assert np.allclose(weights[weights < 0].groupby(level="date").sum(), -1.0)
```

多头权重和为 1, 空头为 -1, 总绝对权重为 2。这里拿到的是目标仓位,
调仓日属于新的持仓区间; `rebalance_dates_belong` 可把共同日历映射到最近已发生的调仓日,
它告诉我们每一天属于哪次调仓, 仓位何时生效还要看实际成交规则。见 [调仓](rebalance.md) 和 [仓位](sizing.md)。

### 4. 单独生成未来标签, 评估分数

```python
from quanttoolkit.metrics import ic_summary, rank_ic
from quanttoolkit.panel.returns import future_returns

gap, horizon = 1, 5
label = future_returns(df=prices, n_periods=horizon, gap=gap)
daily_ic = rank_ic(factor=score, forward_return=label)
summary = ic_summary(ic=daily_ic)

assert label.name == "forward_return_5_gap_1"
assert label.loc[dates[-6:]].isna().all()
assert daily_ic.notna().any()
assert summary.index.tolist() == ["ic_mean", "ic_std", "ic_ir", "ic_pos"]
```

标签在 t 上是 `P[t+6] / P[t+1] - 1`, 假定以 t+1 的所选价格入场, t+6 出场。
它不会改变第 3 步的仓位。这里全区间 IC 只用于演示流程;
实际选因子用训练或验证区间, 固定方案后再看测试区间。
每天计算的 5 期标签会重叠, 不能把这些标签直接当作连续单期收益复利。
见 [收益率](returns.md) 和 [指标](metrics.md)。

## 加模型时: 从因子面板另开一条分支

需要模型打分时, 顺着 **组装样本 → 按日期取训练/验证/测试集 → 训练 → 验证选型 → 测试预测** 往下走。
样本用原始因子列, 标签仍是上面的未来收益。日期分段从完整行情日历生成,
不要先删完缺失样本再把剩余日期当作交易日历。

### 5. 组装样本, 留出标签隔离区间

```python
from quanttoolkit.model.dataset import assemble_samples, label_end_dates
from quanttoolkit.model.split import split_samples

samples = assemble_samples(features=factors, label=label)
parts = split_samples(
    samples=samples,
    dates=calendar,
    label_end=label_end_dates(prices.index, n_periods=horizon, gap=gap),
    train_end=dates[54],
    valid_end=dates[79],
    embargo=gap + horizon,
)
train, valid, test = parts["train"], parts["valid"], parts["test"]

assert all(not part.empty for part in parts.values())
assert train.index.get_level_values("date").max() == dates[48]
assert valid.index.get_level_values("date").min() == dates[55]
assert valid.index.get_level_values("date").max() == dates[73]
assert test.index.get_level_values("date").min() == dates[80]
```

训练和验证段末尾各空出 `gap + horizon = 6` 个共同交易日, 防止标签跨进下一段。
`split_samples` 还会检查标签实际结束日期, 排除证券缺行时仍然跨段的样本。见 [样本](dataset.md) 和 [切分](split.md)。

### 6. 训练模型, 预测后回到同一种分数接口

下面用本包的简化 GBDT 演示回归, 验证集负责挑选树的轮数。

```python
from quanttoolkit.model import SimpleGBDT

feature_cols = factors.columns.tolist()
model = SimpleGBDT(n_trees=10, max_depth=2)
model.fit(
    train[feature_cols].to_numpy(),
    train[label.name].to_numpy(),
    X_val=valid[feature_cols].to_numpy(),
    y_val=valid[label.name].to_numpy(),
    early_stopping_rounds=3,
)
pred = pd.Series(
    model.predict(test[feature_cols].to_numpy()), index=test.index, name="pred"
)
test_summary = ic_summary(ic=rank_ic(factor=pred, forward_return=test[label.name]))
test_calendar = calendar[calendar >= pred.index.get_level_values("date").min()]
test_reb_dates = rebalance_dates(test_calendar, freq=5)
pred_on_reb = pred.loc[pred.index.get_level_values("date").isin(test_reb_dates)]
model_weights = simple_quantile_long_short(score=pred_on_reb, n_quantiles=3)

assert pred.index.equals(test.index)
assert pred.notna().all()
assert np.isfinite(test_summary["ic_mean"])
assert not model_weights.empty
```

`pred` 和手工合成的 `score` 都是 `[date, ticker]` Series, 后面的评估和仓位接口相同。
实际预测只需要当时已经知道的历史特征, 不用等未来标签出来再组装样本; `assemble_samples` 用于离线训练和评估。

换模型时只替换第 6 步:

| 模型 | 怎么调用 |
| --- | --- |
| `RegressionTree` | `fit(X, y)` → `predict(X)` → 包装为带原索引的 Series |
| `SimpleGBDT` | 上面的训练集拟合 + 验证集挑选轮数 → 测试集预测 |
| LightGBM 回归 | `fit_lightgbm` + 验证集早停 → `booster.predict` → 包装为 Series |
| LightGBM 排序 | 训练标签 → `relevance_by_date` → `fit_lambdarank(features=..., relevance=..., params=LambdaRankParams())` → `predict_scores` |

LightGBM 回归传入 `valid_features` 和 `valid_label` 后按验证指标早停。
评估复用 `metrics`, 模型保存直接调用 LightGBM 和 `TrainLog`, 见 [模型说明](model.md)。
排序标签用 `relevance_by_date` 转为 0 到 30 的整数档, 预测返回的是连续分数;
`fit_lambdarank` 内部处理日期分组, 不做早停。各模型细节见 [模型文档](model.md)。

## 替换数据入口

在外部项目使用时, 直接导入 `quanttoolkit` 下的模块。
`scripts/` 是本仓库的运行脚本, 不作为依赖项目的调用入口。

| 数据来源 | 替换第 1 步的调用 |
| --- | --- |
| Yahoo | `from quanttoolkit.sources import get_prices`, 然后 `prices = get_prices(tickers=["AAPL", "MSFT"], start="2020-01-01", end="2025-01-01", progress=False)` |
| 自己的 DuckDB | `from quanttoolkit.db.connection import get_duckdb`, 用 `with get_duckdb(path="data/my_market.db") as connection:` 读取 `SELECT date, ticker, adj_close FROM prices`, `.df()` 后传给 `to_date_ticker_frame(df=raw)` |
| CSV 等长表 | 读成含 `date`, `ticker`, `adj_close` 列的 DataFrame, 再 `to_date_ticker_frame(df=raw.copy())` |

在自己的项目里调用时, 传自己的数据库路径; `quanttoolkit.db.connection.DB` 按本包所在位置推导,
不代表调用方的数据目录。连接默认只读, 写入要显式传 `read_only=False`。

需要缓存 Yahoo 行情时, 顺序是 `get_prices` → 可选 `fill_price_gaps` → `reset_index()` →
`operation_sql(read_sql(SQL_DIR / "prices.sql"), read_sql(SQL_DIR / "prices_ddl.sql"), connection, df)`。
`operation_sql` 内部管理事务。首次获取股票池可调用 `fetch_sp500_universe`, 返回普通列的静态表;
当前成员名单不代表历史成员, 做历史研究时要明确股票池口径。
完整例子见 [数据源](../src/quanttoolkit/sources/USAGE.md) 和 [数据库](../src/quanttoolkit/db/README.md)。

## 仓位之后怎么接

先在自己的项目里把 `weights` 接成实际持仓, 算出逐期简单收益, 再调用 `total_return`, `sharpe`,
`max_drawdown`, `win_rate`。这些指标需要收益序列, 不能直接传入分数或权重, `sharpe` 当前不年化。
持有期重叠的多期收益也不能直接作为连续收益路径交给复利和回撤指标。

基准可以用 `equal_weight_hold(prices=评估区间行情)` 单独构造,
它首日等金额买入, 后续份额不变, 返回单期收益, 首日为 NaN。
调用收益指标前可显式移除这个没有前期价格的首日, 中间缺失收益应先查原因。
比较策略和基准时对齐时间区间与收益频率。见 [基准](benchmark.md) 和 [指标](metrics.md)。

需要记录自己的流程时, 在外部项目的步骤函数上加 `@trace`, 用 `print_call_chain()` 查看运行顺序,
见 [函数追踪](tracelog.md)。
