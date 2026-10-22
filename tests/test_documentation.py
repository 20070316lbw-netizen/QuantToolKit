import doctest
import re
import runpy
from pathlib import Path

import pytest

from quanttoolkit import metrics
from quanttoolkit.model import dataset, split
from quanttoolkit.model import lambdarank as model_lambdarank
from quanttoolkit.model import lightgbm as model_lightgbm
from quanttoolkit.model import params as model_params
from quanttoolkit.model import relevance as model_relevance
from quanttoolkit.panel import returns, transfer, volatility
from quanttoolkit.portfolio import benchmark, rebalance

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "returns",
        "volatility",
        "data-structure",
        "rebalance",
        "factors",
        "split",
        "dataset",
        "model",
        "metrics",
        "sizing",
        "benchmark",
    ],
)
def test_offline_documentation_examples(name):
    source = (ROOT / "docs" / f"{name}.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.DOTALL)
    assert blocks, "文档应包含可执行示例"
    for block in blocks:
        # 仅执行仓库中明确列出的离线文档示例, 以校验其断言和 API 契约
        exec(compile(block, f"docs/{name}.md", "exec"), {})  # noqa: S102


@pytest.mark.parametrize(
    "module",
    [
        returns,
        volatility,
        rebalance,
        transfer,
        metrics,
        benchmark,
        split,
        dataset,
        model_params,
        model_lightgbm,
        model_lambdarank,
        model_relevance,
    ],
)
def test_calculation_docstrings(module):
    result = doctest.testmod(module)
    assert result.failed == 0
    assert result.attempted > 0


def test_offline_example_script(capsys):
    runpy.run_path(
        str(ROOT / "scripts" / "example_price_metrics.py"), run_name="__main__"
    )
    output = capsys.readouterr().out
    for name in [
        "historical_return_2",
        "forward_return_2_gap_1",
        "log_return",
        "volatility_2",
        "forward_volatility_2_gap_1",
        "rebalance_date",
    ]:
        assert name in output
