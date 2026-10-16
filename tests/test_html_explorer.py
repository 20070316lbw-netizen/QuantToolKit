"""核对交互说明中的浏览器计算与实际 Python 接口"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quanttoolkit.factors.mom_base import base_momfactor
from quanttoolkit.factors.reversal_base import base_reversalfactor
from quanttoolkit.returns import future_returns, historical_return
from quanttoolkit.volatility import forward_volatility, history_vol

EXPLORER = Path(__file__).resolve().parents[1] / "docs" / "explorer.html"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="浏览器计算回归测试需要 Node.js")
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize(
    "mode, function",
    [
        ("historical", historical_return),
        ("future", future_returns),
        ("historyVol", history_vol),
        ("forwardVol", forward_volatility),
        ("momentum", base_momfactor),
        ("reversal", base_reversalfactor),
    ],
)
def test_explorer_matches_python_for_all_available_windows(mode, function, missing):
    source = EXPLORER.read_text()
    calculator = re.search(
        r'<script id="calculator">(.*?)</script>', source, re.DOTALL
    )[1]
    runner = (
        calculator
        + """
const mode = process.argv[1];
const missing = process.argv[2] === 'true';
const prices = samplePrices.map((p, i) => missing && i === 8 ? null : p);
const results = [];
for (let window = 2; window <= 5; window++) {
  for (let offset = 0; offset <= 3; offset++) {
    results.push(Array.from({length: 12}, (_, t) =>
      calculate(prices, mode, t, window, offset).value));
  }
}
process.stdout.write(JSON.stringify({prices, results}));
"""
    )
    browser = json.loads(
        subprocess.run(
            [NODE, "-e", runner, mode, str(missing).lower()],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-02", periods=12), ["A"]],
        names=["date", "ticker"],
    )
    panel = pd.DataFrame({"adj_close": browser["prices"]}, index=index, dtype=float)
    expected = []
    for window in range(2, 6):
        for offset in range(4):
            if mode in {"momentum", "reversal"}:
                result = function(panel, window=window, skip=offset)
            else:
                kwargs = {"df": panel, "price_col": "adj_close"}
                kwargs[
                    "n_periods" if mode in {"historical", "future"} else "window"
                ] = window
                if mode in {"future", "forwardVol"}:
                    kwargs["gap"] = offset
                result = function(**kwargs)
            expected.append(result.to_numpy())
    np.testing.assert_allclose(
        np.asarray(browser["results"], dtype=float),
        expected,
        atol=1e-12,
        rtol=1e-10,
        equal_nan=True,
    )
