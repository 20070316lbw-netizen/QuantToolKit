"""绘制量化研究与回测图表, 也可以按流程拼接报告"""

from .plots import (
    plot_drawdown,
    plot_execution_slippage,
    plot_forward_spread,
    plot_ic,
    plot_nav,
    plot_target_exposure,
)
from .report import plot_report

__all__ = [
    "plot_drawdown",
    "plot_execution_slippage",
    "plot_forward_spread",
    "plot_ic",
    "plot_nav",
    "plot_report",
    "plot_target_exposure",
]
