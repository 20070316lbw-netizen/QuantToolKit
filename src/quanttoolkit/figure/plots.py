"""把研究和回测产物画在独立坐标轴上, 单图也可以放进组合报告"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter

from quanttoolkit.registry import _contract

_BLUE = "#2457a6"
_RED = "#a63d4a"
_TEAL = "#19766d"


def _numeric(values: pd.Series, name: str, *, allow_nan: bool = False) -> pd.Series:
    if not pd.api.types.is_any_real_numeric_dtype(values.dtype):
        raise ValueError(f"{name} 必须是数值 Series")
    values = values.astype(float)
    invalid = np.isinf(values) if allow_nan else ~np.isfinite(values)
    if invalid.any():
        raise ValueError(f"{name} 包含无效数值")
    return values


def _dated(
    values: pd.Series,
    name: str,
    *,
    allow_nan: bool = False,
    per_fill: bool = False,
) -> pd.Series:
    if not isinstance(values, pd.Series) or not isinstance(
        values.index, pd.DatetimeIndex
    ):
        raise TypeError(f"{name} 必须是以 DatetimeIndex 为索引的 Series")
    if values.index.hasnans or (not per_fill and not values.index.is_unique):
        raise ValueError(f"{name} 日期不能缺失, 非逐笔数据的日期必须唯一")
    if values.empty and not per_fill:
        raise ValueError(f"{name} 不能为空")
    return _numeric(values, name, allow_nan=allow_nan).sort_index()


def _axis(
    ax: Axes | None,
    dates: pd.DatetimeIndex,
    title: str,
    ylabel: str,
    *,
    percent: bool = False,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3), layout="constrained")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dce3ed", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if not dates.empty:
        locator = mdates.AutoDateLocator(tz=dates.tz, minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=dates.tz))
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    return ax


def _reference(ax: Axes, level: float = 0) -> None:
    ax.axhline(level, color="#66768a", linewidth=0.9, linestyle="--", zorder=1)


def _nav(portfolio_returns: pd.Series) -> pd.Series:
    returns = _dated(portfolio_returns, "portfolio_returns")
    return (1 + returns).cumprod()


@_contract
def plot_nav(portfolio_returns: pd.Series, *, ax: Axes | None = None) -> Axes:
    """从初始净值 1 复利, 绘制组合净值

    Args:
        portfolio_returns: 已登记的 DatetimeIndex Series, 每期一条连续且非重叠的组合简单收益
            外部回测收益用 prepare_portfolio_returns 接入
            值使用小数收益率, 日期唯一且不能有缺失收益, 按日期排序后计算
        ax: 默认 None, 新建坐标轴, 也可传入已有坐标轴

    Returns:
        Axes, 每个日期显示计入当期收益后的净值, 虚线表示初始净值 1

    Raises:
        TypeError: 输入不是以 DatetimeIndex 为索引的 Series
        ValueError: 收益来源不符, 为空, 日期重复或缺失, 或存在非数值及非有限收益

    Example:
        >>> from quanttoolkit.metrics import prepare_portfolio_returns
        >>> returns = pd.Series([0.1, 0.0], index=pd.date_range("2024-01-01", periods=2))
        >>> ax = plot_nav(prepare_portfolio_returns(returns))
        >>> ax.lines[0].get_ydata().tolist()
        [1.1, 1.1]
        >>> plt.close(ax.figure)
    """
    nav = _nav(portfolio_returns)
    ax = _axis(ax, nav.index, "Portfolio NAV", "NAV (start = 1)")
    ax.plot(nav.index, nav, color=_BLUE, marker=".", linewidth=1.7)
    _reference(ax, 1)
    return ax


@_contract
def plot_drawdown(portfolio_returns: pd.Series, *, ax: Axes | None = None) -> Axes:
    """绘制相对历史峰值的组合回撤, 初始峰值为 1

    Args:
        portfolio_returns: 已登记的 DatetimeIndex Series, 每期一条连续且非重叠的组合简单收益
            外部回测收益用 prepare_portfolio_returns 接入
            值使用小数收益率, 日期唯一且不能有缺失收益, 按日期排序后计算
        ax: 默认 None, 新建坐标轴, 也可传入已有坐标轴

    Returns:
        Axes, 纵轴为净值 / 历史峰值 - 1, 显示为百分比, 首期亏损也计入

    Raises:
        TypeError: 输入不是以 DatetimeIndex 为索引的 Series
        ValueError: 收益来源不符, 为空, 日期重复或缺失, 或存在非数值及非有限收益

    Example:
        >>> from quanttoolkit.metrics import prepare_portfolio_returns
        >>> returns = pd.Series([-0.25], index=pd.date_range("2024-01-01", periods=1))
        >>> ax = plot_drawdown(prepare_portfolio_returns(returns))
        >>> ax.lines[0].get_ydata().tolist()
        [-0.25]
        >>> plt.close(ax.figure)
    """
    nav = _nav(portfolio_returns)
    drawdown = nav.div(nav.cummax().clip(lower=1)).sub(1)
    ax = _axis(ax, nav.index, "Portfolio drawdown", "Drawdown", percent=True)
    ax.fill_between(nav.index, drawdown, 0, color=_RED, alpha=0.15)
    ax.plot(nav.index, drawdown, color=_RED, marker=".", linewidth=1.4)
    _reference(ax)
    return ax


@_contract
def plot_target_exposure(target_weights: pd.Series, *, ax: Axes | None = None) -> Axes:
    """按调仓日绘制目标权重的总敞口和净敞口

    Args:
        target_weights: [date, ticker] MultiIndex Series, 每个键唯一, date 为日期类型
            权重使用小数, 空头为负, 无缺失值, 例如 simple_quantile_long_short 的结果
        ax: 默认 None, 新建坐标轴, 也可传入已有坐标轴

    Returns:
        Axes, 总敞口为当日权重绝对值之和, 净敞口为当日权重之和
        只使用输入里的调仓日, 纵轴显示百分比

    Raises:
        TypeError: 权重不是 Series
        ValueError: 权重为空, 面板索引无效, 或存在非数值及非有限权重

    Example:
        >>> dates = pd.date_range("2024-01-01", periods=1)
        >>> index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
        >>> ax = plot_target_exposure(pd.Series([1.0, -1.0], index=index))
        >>> [line.get_ydata()[0].item() for line in ax.lines[:2]]
        [2.0, 0.0]
        >>> plt.close(ax.figure)
    """
    if not isinstance(target_weights, pd.Series):
        raise TypeError("target_weights 必须是 [date, ticker] Series")
    index = target_weights.index
    if (
        not isinstance(index, pd.MultiIndex)
        or index.names != ["date", "ticker"]
        or not index.is_unique
        or not isinstance(index.levels[0], pd.DatetimeIndex)
        or any(-1 in codes for codes in index.codes)
        or target_weights.empty
    ):
        raise ValueError("target_weights 必须具有非空且唯一的 [date, ticker] 日期键")
    weights = _numeric(target_weights, "target_weights")
    gross = weights.abs().groupby(level="date").sum()
    net = weights.groupby(level="date").sum()
    ax = _axis(ax, gross.index, "Target exposure", "Exposure", percent=True)
    ax.plot(gross.index, gross, color=_BLUE, marker=".", label="Gross")
    ax.plot(net.index, net, color=_TEAL, marker=".", label="Net")
    _reference(ax)
    ax.legend(loc="best", frameon=False)
    return ax


@_contract
def plot_execution_slippage(slippage_bps: pd.Series, *, ax: Axes | None = None) -> Axes:
    """绘制每笔成交相对于参考价的滑点

    Args:
        slippage_bps: 以成交时间为 DatetimeIndex 的 Series, 同一时间可有多笔
            由回测调用方算好, 单位为基点, 正值表示成交更贵或卖出更便宜
        ax: 默认 None, 新建坐标轴, 也可传入已有坐标轴

    Returns:
        Axes, 每笔成交一个点, 空 Series 显示 No fills

    Raises:
        TypeError: 输入不是以 DatetimeIndex 为索引的 Series
        ValueError: 成交时间缺失, 或存在非数值及非有限滑点

    Example:
        >>> times = pd.to_datetime(["2024-01-01 09:30", "2024-01-01 09:30"])
        >>> ax = plot_execution_slippage(pd.Series([1.5, -0.5], index=times))
        >>> ax.collections[0].get_offsets()[:, 1].tolist()
        [1.5, -0.5]
        >>> plt.close(ax.figure)
    """
    slippage = _dated(slippage_bps, "slippage_bps", per_fill=True)
    ax = _axis(ax, slippage.index, "Execution slippage", "Slippage (bps)")
    _reference(ax)
    if slippage.empty:
        ax.text(0.5, 0.65, "No fills", ha="center", transform=ax.transAxes)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    else:
        ax.scatter(slippage.index, slippage, color=_TEAL, s=22, alpha=0.75)
    return ax


@_contract
def plot_ic(ic: pd.Series, *, ax: Axes | None = None) -> Axes:
    """绘制截面 IC 或 Rank IC 时序

    Args:
        ic: DatetimeIndex Series, 每天一个相关系数, 日期唯一
            例如 ic 或 rank_ic 的结果, 允许 NaN, 绘图时保留缺口
        ax: 默认 None, 新建坐标轴, 也可传入已有坐标轴

    Returns:
        Axes, 纵轴为无量纲相关系数, 虚线为 0

    Raises:
        TypeError: 输入不是以 DatetimeIndex 为索引的 Series
        ValueError: 序列为空, 日期重复或缺失, 或存在非数值及无穷值

    Example:
        >>> values = pd.Series([0.3], index=pd.date_range("2024-01-01", periods=1))
        >>> ax = plot_ic(values)
        >>> ax.lines[0].get_ydata().tolist()
        [0.3]
        >>> plt.close(ax.figure)
    """
    daily = _dated(ic, "ic", allow_nan=True)
    title = "Daily Rank IC" if daily.name == "rank_ic" else "Daily IC"
    ax = _axis(ax, daily.index, title, "Correlation")
    ax.plot(daily.index, daily, color=_BLUE, marker=".", linewidth=1.3)
    _reference(ax)
    return ax


@_contract
def plot_forward_spread(forward_spread: pd.Series, *, ax: Axes | None = None) -> Axes:
    """按信号日期绘制每次观测的远期多空收益差

    Args:
        forward_spread: DatetimeIndex Series, 每个信号日一个远期多空收益差
            值使用小数收益率, 允许持有期重叠, NaN 保留为缺口
        ax: 默认 None, 新建坐标轴, 也可传入已有坐标轴

    Returns:
        Axes, 纵轴显示每次观测的收益差百分比, 各期不复利

    Raises:
        TypeError: 输入不是以 DatetimeIndex 为索引的 Series
        ValueError: 序列为空, 日期重复或缺失, 或存在非数值及无穷值

    Example:
        >>> spread = pd.Series([0.1, -0.05], index=pd.date_range("2024-01-01", periods=2))
        >>> ax = plot_forward_spread(spread)
        >>> ax.lines[0].get_ydata().tolist()
        [0.1, -0.05]
        >>> plt.close(ax.figure)
    """
    spread = _dated(forward_spread, "forward_spread", allow_nan=True)
    ax = _axis(ax, spread.index, "Forward long-short spread", "Spread", percent=True)
    ax.plot(spread.index, spread, color=_TEAL, marker=".", linewidth=1.3)
    _reference(ax)
    return ax
