"""固定各流程的图表组合, 用已有产物拼成一张报告"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .plots import (
    plot_drawdown,
    plot_execution_slippage,
    plot_forward_spread,
    plot_ic,
    plot_nav,
    plot_target_exposure,
)

_PERFORMANCE_PLOTS = (
    ("portfolio_returns", plot_nav),
    ("portfolio_returns", plot_drawdown),
)

# 每项登记必需图、附加图和报告标题, 产物名同时用作 plot_report 的参数名
_REPORTS = {
    "vectorized": (
        _PERFORMANCE_PLOTS,
        (("target_weights", plot_target_exposure),),
        "Vectorized backtest",
    ),
    "event_driven": (
        _PERFORMANCE_PLOTS,
        (("slippage_bps", plot_execution_slippage),),
        "Event-driven backtest",
    ),
    "factor_research": (
        (("ic", plot_ic),),
        (("forward_spread", plot_forward_spread),),
        "Factor research",
    ),
}


def plot_report(
    kind: Literal["vectorized", "event_driven", "factor_research"],
    *,
    output: str | Path | None = None,
    **artifacts: pd.Series,
) -> Figure:
    """按流程拼图, 有附加产物时补上该流程的诊断图

    Args:
        kind: vectorized, event_driven 或 factor_research
        output: 默认 None, 只返回图, 也可以传 PNG 或 PDF 等文件路径直接保存
        **artifacts: 两种回测均必传已登记的 portfolio_returns, 用于净值和回撤
            vectorized 可选 target_weights, event_driven 可选 slippage_bps
            factor_research 必传 ic, 可选 forward_spread
            各产物格式和单位见对应的单图函数

    Returns:
        Figure, 每个图占一行, 顺序按流程固定

    Raises:
        TypeError: 产物类型不符
        ValueError: 流程未知, 缺少必需产物, 产物不属于当前流程, 或格式无效
        OSError: 不能保存到指定路径

    Example:
        >>> from quanttoolkit.metrics import prepare_portfolio_returns
        >>> returns = pd.Series([0.1, -0.05], index=pd.date_range("2024-01-01", periods=2))
        >>> figure = plot_report("vectorized", portfolio_returns=prepare_portfolio_returns(returns))
        >>> len(figure.axes)
        2
        >>> plt.close(figure)
    """
    if kind not in _REPORTS:
        raise ValueError(f"未知出图流程 {kind!r}, 可选 {', '.join(_REPORTS)}")
    required, optional, title = _REPORTS[kind]
    required_names = {name for name, _ in required}
    allowed = required_names | {name for name, _ in optional}
    missing = required_names - artifacts.keys()
    unexpected = artifacts.keys() - allowed
    if missing:
        raise ValueError(f"{kind} 报告缺少产物: {', '.join(sorted(missing))}")
    if unexpected:
        raise ValueError(f"{kind} 报告不接受产物: {', '.join(sorted(unexpected))}")

    plots = list(required)
    plots.extend((name, plot) for name, plot in optional if name in artifacts)
    figure, axes = plt.subplots(
        len(plots),
        1,
        sharex=True,
        squeeze=False,
        figsize=(10, 1 + 2.6 * len(plots)),
        layout="constrained",
    )
    figure.suptitle(title, fontsize=16, fontweight="bold")
    try:
        last_data_ax = axes[0, 0]
        for (name, plot), ax in zip(plots, axes[:, 0], strict=True):
            values = artifacts[name]
            plot(values, ax=ax)
            if not values.empty:
                last_data_ax = ax
        # 最后一张有数据的图保留时间标签, 无成交的诊断图不占日期轴
        last_data_ax.tick_params(axis="x", labelbottom=True)
        if output is not None:
            figure.savefig(output, dpi=150)
    except Exception:
        plt.close(figure)
        raise
    return figure
