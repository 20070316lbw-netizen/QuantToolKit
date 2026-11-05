"""查询公开接口的契约, 在已接入的函数入口检查格式与上游"""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
from inspect import signature
from threading import Lock
from weakref import ref

import numpy as np
import pandas as pd

from quanttoolkit._catalog import CATALOG

__all__ = ["get_contract", "list_contracts"]

_origins = {}
_origin_lock = Lock()


def get_contract(name: str) -> dict:
    """查询单个接口, 名称不含 quanttoolkit 前缀, 返回契约副本"""
    return deepcopy(CATALOG[name])


def list_contracts(*, stage: str | None = None, workflow: str | None = None) -> dict:
    """按接口名称查询契约, 可筛选所属阶段和明确登记的流程"""
    return deepcopy(
        {
            name: spec
            for name, spec in CATALOG.items()
            if (stage is None or spec["stage"] == stage)
            and (workflow is None or workflow in spec["workflows"])
        }
    )


def _origin(value):
    with _origin_lock:
        item = _origins.get(id(value))
        if item is not None and item[0]() is value:
            return item[1], item[2]
    return None


def _remember(value, name, metadata):
    key = id(value)

    def forget(reference):
        with _origin_lock:
            item = _origins.get(key)
            if item is not None and item[0] is reference:
                del _origins[key]

    reference = ref(value, forget)
    with _origin_lock:
        _origins[key] = reference, name, metadata


def _panel_index(index, context):
    if not isinstance(index, pd.MultiIndex):
        raise TypeError(f"{context} 必须有 [date, ticker] MultiIndex")
    if index.names != ["date", "ticker"] or not isinstance(
        index.levels[0], pd.DatetimeIndex
    ):
        raise ValueError(f"{context} 索引必须是 [date, ticker], date 层须为日期类型")
    if not index.is_unique or any(-1 in codes for codes in index.codes):
        raise ValueError(f"{context} 的 [date, ticker] 键必须唯一且无缺失")
    tickers = index.levels[1].take(np.unique(index.codes[1]))
    if any(not isinstance(ticker, str) or not ticker.strip() for ticker in tickers):
        raise ValueError(f"{context} 的 ticker 必须是非空证券代码")


def _panel_frame(frame, context):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{context} 必须是 DataFrame")
    _panel_index(frame.index, context)


def _check_price(frame, context, price_col):
    _panel_frame(frame, context)
    if price_col not in frame.columns:
        raise KeyError(f"{context} 缺少价格列 {price_col!r}")
    prices = frame[price_col]
    if not isinstance(prices, pd.Series) or (
        not frame.empty and not pd.api.types.is_any_real_numeric_dtype(prices.dtype)
    ):
        raise ValueError(f"{context} 的价格列 {price_col!r} 必须是唯一的数值列")
    values = prices.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any() or (values <= 0).any():
        raise ValueError(f"{context} 的非缺失价格必须有限且大于 0")


def _label_end(values, context):
    if not isinstance(values, pd.Series):
        raise TypeError(f"{context} 必须是 Series")
    _panel_index(values.index, context)
    if not pd.api.types.is_datetime64_any_dtype(values.dtype):
        raise ValueError(f"{context} 的值必须是日期, 可包含 NaT")


def _panel_numeric(values, context):
    if not isinstance(values, pd.Series):
        raise TypeError(f"{context} 必须是数值 Series")
    _panel_index(values.index, context)
    if not values.empty and not pd.api.types.is_any_real_numeric_dtype(values.dtype):
        raise ValueError(f"{context} 必须是数值 Series")


def _dated_numeric(values, context, *, finite=False):
    if not isinstance(values, pd.Series) or not isinstance(
        values.index, pd.DatetimeIndex
    ):
        raise TypeError(f"{context} 必须是以 DatetimeIndex 为索引的 Series")
    if (finite and values.empty) or values.index.hasnans or not values.index.is_unique:
        raise ValueError(f"{context} 日期必须唯一且无缺失, 组合收益不能为空")
    if not values.empty and not pd.api.types.is_any_real_numeric_dtype(values.dtype):
        raise ValueError(f"{context} 必须是数值 Series")
    numbers = values.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(numbers).any() or (finite and np.isnan(numbers).any()):
        raise ValueError(f"{context} 包含无效数值")


_CHECKS = {
    "panel_index": _panel_index,
    "panel_frame": _panel_frame,
    "panel_numeric": _panel_numeric,
    "label_end": _label_end,
    "dated_numeric": _dated_numeric,
}


def _check_input(name, arg, values, rule):
    value = values[arg]
    context = f"{name}.{arg}"
    policy = rule["source_policy"]
    origin = None
    if policy != "described":
        origin = _origin(value)
        if origin is not None and origin[0] not in rule["upstream"]:
            allowed = ", ".join(rule["upstream"]) or "相应的外部数据"
            raise ValueError(f"{context} 不接受 {origin[0]} 的输出, 允许 {allowed}")

    check = rule["check"]
    if check == "price":
        _check_price(value, context, values["price_col"])
    elif check == "dated_finite":
        _dated_numeric(value, context, finite=True)
    elif check is not None:
        _CHECKS[check](value, context)
    if origin is None and policy == "required":
        raise ValueError(f"{context} 缺少上游, 请先调用 {', '.join(rule['upstream'])}")


def _contract(function):
    name = (
        function.__module__.removeprefix("quanttoolkit.") + "." + function.__qualname__
    )
    spec = CATALOG[name]
    bind = signature(function).bind
    checks = {
        arg: rule
        for arg, rule in spec["inputs"].items()
        if rule["check"] is not None or rule["source_policy"] != "described"
    }

    @wraps(function)
    def checked(*args, **kwargs):
        bound = bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        for arg, rule in checks.items():
            _check_input(name, arg, values, rule)
        result = function(*args, **kwargs)
        if spec["track"]:
            metadata = {key: values[arg] for key, arg in spec["capture"].items()}
            if spec["inherit"] is not None:
                origin = _origin(values[spec["inherit"]])
                if origin is not None:
                    metadata.update(origin[1])
            _remember(result, name, metadata)
        return result

    return checked
