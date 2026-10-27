import doctest
import re
import runpy
from pathlib import Path

import lightgbm as lgb
import pytest

from quanttoolkit import metrics
from quanttoolkit.model import TrainLog, dataset, split
from quanttoolkit.model import gbdt as model_gbdt
from quanttoolkit.model import lambdarank as model_lambdarank
from quanttoolkit.model import lightgbm as model_lightgbm
from quanttoolkit.model import params as model_params
from quanttoolkit.model import relevance as model_relevance
from quanttoolkit.model import tree as model_tree
from quanttoolkit.panel import returns, transfer, volatility
from quanttoolkit.portfolio import benchmark, rebalance

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTATION = [
    ROOT / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
    ROOT / "src/quanttoolkit/db/README.md",
    ROOT / "src/quanttoolkit/sources/USAGE.md",
]


@pytest.mark.parametrize("path", DOCUMENTATION, ids=lambda path: path.stem)
def test_local_documentation_links(path):
    source = path.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", source):
        if target.startswith(("https://", "http://", "mailto:")):
            continue
        local_path = target.split("#", 1)[0]
        if local_path:
            assert (path.parent / local_path).exists(), f"{path}: {target}"


@pytest.mark.parametrize("section", ["单独写入成员名单", "写入日行情"])
def test_database_documentation_examples(section):
    source = (ROOT / "src/quanttoolkit/db/README.md").read_text(encoding="utf-8")
    content = source.split(f"## {section}\n", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
    assert blocks, "数据库文档应包含可独立执行的内存数据库示例"
    for block in blocks:
        exec(compile(block, f"db/README.md:{section}", "exec"), {})  # noqa: S102


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
        "tracelog",
    ],
)
def test_offline_documentation_examples(name):
    source = (ROOT / "docs" / f"{name}.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.DOTALL)
    assert blocks, "文档应包含可执行示例"
    for block in blocks:
        # 只运行上面列出的离线示例, 检查调用方式和预期结果
        exec(compile(block, f"docs/{name}.md", "exec"), {})  # noqa: S102


def test_calling_guide_workflow():
    source = (ROOT / "docs" / "calling-guide.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.DOTALL)
    assert blocks, "调用指南应包含可执行的完整流程"
    # 后面的步骤要用到前面的变量, 按顺序一起运行才能检查接口是否接得上
    exec(compile("\n".join(blocks), "docs/calling-guide.md", "exec"), {})  # noqa: S102


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
        model_tree,
        model_gbdt,
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


def test_offline_model_example(tmp_path):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "example_model_workflow.py"
    )
    main = runpy.run_path(str(script))["main"]
    output = main(tmp_path / "example")
    for task in ("regression", "ranking"):
        loaded = lgb.Booster(model_file=output / task / "model.txt")
        log = TrainLog.load(output / task / "run.json")
        assert isinstance(loaded, lgb.Booster)
        assert log.metrics["test"]["n_samples"] > 0
        assert ("mse" in log.metrics["test"]) == (task == "regression")
        assert TrainLog.load(output / task / "run.json") == log
