"""复现首轮失败后追加的隔夜反转探索, 不重新选择窗口与方向"""

import json

import experiment


def main():
    origin = experiment.HERE
    out = origin / "followup_overnight"
    out.mkdir(exist_ok=True)
    record = {
        "factor": "overnight_reversal_63",
        "formula": "-sum(log((open[t] * adj_close[t] / close[t]) / adj_close[t-1]), 63)",
        "selection_reason": (
            "Best minimum Rank IC across discovery and selection "
            "among candidates positive in both periods"
        ),
        "timing": "Chosen after seeing primary low-volatility failure in check and holdout",
        "status": "Exploratory second look, not untouched out-of-sample confirmation",
        "windows_or_signs_changed": False,
        "comparison_mask": "Original eight-factor common mask, unchanged",
        "no_further_candidate_switches": True,
    }
    (out / "followup_protocol.json").write_text(json.dumps(record, indent=2) + "\n")
    experiment.HERE = out
    prices, industry = experiment._load_prices()
    factors = experiment._make_factors(prices)
    summary, uncertainty = experiment._evaluate(
        prices, factors, industry, "overnight_reversal_63"
    )
    print(summary.to_string(index=False))
    print(uncertainty.to_string(index=False))


if __name__ == "__main__":
    main()
