"""把时点科目和期间科目收成申报日面板"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

EQUITY_CONCEPTS = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
REVENUE_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
)
INCOME_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")
SHARES_CONCEPT = "EntityCommonStockSharesOutstanding"

QUARTER_DAYS = (80, 100)
ANNUAL_DAYS = (350, 380)
GROWTH_GAP_DAYS = (300, 430)
GROWTH_TARGET_DAYS = 365

PANEL_COLUMNS = [
    "ticker",
    "filed",
    "assets",
    "equity",
    "shares",
    "revenue_q",
    "net_income_q",
    "net_income_ttm",
    "asset_growth",
]
LEVEL_COLUMNS = [
    "assets",
    "equity",
    "shares",
    "revenue_q",
    "net_income_q",
    "net_income_ttm",
]

__all__ = [
    "EQUITY_CONCEPTS",
    "INCOME_CONCEPTS",
    "PANEL_COLUMNS",
    "REVENUE_CONCEPTS",
    "SHARES_CONCEPT",
    "align_filings",
    "filing_panel",
    "read_filing_panel",
]


def _sql_in(concepts: tuple[str, ...]) -> str:
    return ", ".join(f"'{concept}'" for concept in concepts)


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _with_ticker(facts: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    keys = members.loc[:, ["ticker", "cik"]].copy()
    keys["cik"] = keys["cik"].astype(str)
    frame = facts.copy()
    frame["cik"] = frame["cik"].astype(str)
    return frame.merge(keys, on="cik", how="inner")


def _prefer(concepts: tuple[str, ...]) -> dict[str, int]:
    return {concept: -index for index, concept in enumerate(concepts)}


def _coerce_dates(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_datetime(frame[column])
    frame["val"] = pd.to_numeric(frame["val"])
    return frame


def _dedup(frame: pd.DataFrame, concepts: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.copy()
    ranked["prefer"] = ranked["concept"].map(_prefer(concepts))
    ranked["accn"] = ranked["accn"].astype(str)
    ranked = ranked.sort_values(
        ["ticker", "period_end", "filed", "prefer", "accn"],
        kind="mergesort",
    )
    return ranked.drop_duplicates(["ticker", "period_end", "filed"], keep="last")


def _instant_points(
    instant: pd.DataFrame,
    members: pd.DataFrame,
    concepts: tuple[str, ...],
    taxonomy: str,
    unit: str,
) -> pd.DataFrame:
    columns = ["ticker", "period_end", "filed", "val"]
    if instant.empty:
        return _empty(columns)
    mask = (
        instant["concept"].isin(concepts)
        & (instant["taxonomy"] == taxonomy)
        & (instant["unit"] == unit)
    )
    frame = instant.loc[mask, ["cik", "concept", "period_end", "filed", "val", "accn"]]
    if frame.empty:
        return _empty(columns)
    frame = _coerce_dates(_with_ticker(frame, members), ("period_end", "filed"))
    frame = frame.dropna(subset=["period_end", "filed", "val"])
    frame = _dedup(frame, concepts)
    return frame.loc[:, columns]


def _period_points(
    period: pd.DataFrame,
    members: pd.DataFrame,
    concepts: tuple[str, ...],
    bounds: tuple[int, int],
) -> pd.DataFrame:
    columns = ["ticker", "period_start", "period_end", "filed", "val"]
    if period.empty:
        return _empty(columns)
    mask = (
        period["concept"].isin(concepts)
        & (period["taxonomy"] == "us-gaap")
        & (period["unit"] == "USD")
    )
    frame = period.loc[
        mask,
        ["cik", "concept", "period_start", "period_end", "filed", "val", "accn"],
    ]
    if frame.empty:
        return _empty(columns)
    frame = _coerce_dates(
        _with_ticker(frame, members),
        ("period_start", "period_end", "filed"),
    )
    frame = frame.dropna(subset=["period_start", "period_end", "filed", "val"])
    span = (frame["period_end"] - frame["period_start"]).dt.days
    frame = frame.loc[span.between(bounds[0], bounds[1])]
    frame = _dedup(frame, concepts)
    return frame.loc[:, columns]


def _known(points: pd.DataFrame) -> pd.DataFrame:
    """每个披露日上, 各 period_end 已披露修订里 filed 最晚的 val"""

    columns = ["ticker", "filed", "period_end", "val"]
    if points.empty:
        return _empty(columns)
    events = points.loc[:, ["ticker", "filed"]].drop_duplicates()
    known = events.merge(points, on="ticker", suffixes=("", "_src"))
    known = known.loc[known["filed_src"] <= known["filed"]]
    known = known.sort_values("filed_src", kind="mergesort").drop_duplicates(
        ["ticker", "filed", "period_end"],
        keep="last",
    )
    return known.loc[:, columns]


def _current(points: pd.DataFrame, name: str) -> pd.DataFrame:
    known = _known(points)
    if known.empty:
        return _empty(["ticker", "filed", name])
    current = (
        known.sort_values("period_end", kind="mergesort")
        .groupby(["ticker", "filed"], sort=False)
        .tail(1)
    )
    return current.rename(columns={"val": name}).loc[:, ["ticker", "filed", name]]


def _asset_series(points: pd.DataFrame) -> pd.DataFrame:
    known = _known(points)
    columns = ["ticker", "filed", "assets", "asset_growth"]
    if known.empty:
        return _empty(columns)
    current = (
        known.sort_values("period_end", kind="mergesort")
        .groupby(["ticker", "filed"], sort=False)
        .tail(1)
        .rename(columns={"period_end": "period_end_cur", "val": "assets"})
    )
    prior = known.merge(
        current.loc[:, ["ticker", "filed", "period_end_cur"]],
        on=["ticker", "filed"],
    )
    gap = (prior["period_end_cur"] - prior["period_end"]).dt.days
    prior = prior.loc[gap.between(GROWTH_GAP_DAYS[0], GROWTH_GAP_DAYS[1])].copy()
    prior["dist"] = (
        (prior["period_end_cur"] - prior["period_end"])
        .dt.days.sub(GROWTH_TARGET_DAYS)
        .abs()
    )
    prior = (
        prior.sort_values(
            ["dist", "period_end"],
            ascending=[True, False],
            kind="mergesort",
        )
        .groupby(["ticker", "filed"], sort=False)
        .head(1)
    )
    growth = current.merge(
        prior.loc[:, ["ticker", "filed", "val"]],
        on=["ticker", "filed"],
        how="left",
    )
    base = growth["val"]
    growth["asset_growth"] = (growth["assets"] / base - 1).where(base != 0)
    return growth.loc[:, columns]


def _derive_q4(quarterly: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """全年截止日没有单季时, 用全年减去该财年内已经披露的前三季"""

    columns = ["ticker", "period_end", "filed", "val", "prefer"]
    if quarterly.empty or annual.empty:
        return _empty(columns)
    quarters = quarterly.rename(
        columns={"period_end": "q_end", "filed": "q_filed", "val": "q_val"}
    )
    attached = annual.merge(
        quarters.loc[:, ["ticker", "q_end", "q_filed", "q_val"]],
        on="ticker",
        how="left",
    )
    known = attached.loc[
        attached["q_filed"].notna() & (attached["q_filed"] <= attached["filed"])
    ]
    standalone = known.loc[
        known["q_end"] == known["period_end"],
        ["ticker", "period_end", "filed"],
    ].drop_duplicates()
    inside = known.loc[
        (known["q_end"] > known["period_start"])
        & (known["q_end"] < known["period_end"])
    ]
    inside = inside.sort_values("q_filed", kind="mergesort").drop_duplicates(
        ["ticker", "period_end", "filed", "q_end"],
        keep="last",
    )
    summed = inside.groupby(["ticker", "period_end", "filed"], as_index=False).agg(
        n=("q_end", "nunique"),
        qsum=("q_val", "sum"),
    )
    summed = summed.loc[summed["n"] == 3]
    derived = annual.merge(summed, on=["ticker", "period_end", "filed"], how="inner")
    if not standalone.empty:
        derived = derived.merge(
            standalone,
            on=["ticker", "period_end", "filed"],
            how="left",
            indicator=True,
        )
        derived = derived.loc[derived["_merge"] == "left_only"]
    derived["val"] = derived["val"] - derived["qsum"]
    derived["prefer"] = 0
    return derived.loc[:, columns]


def _one_val(points: pd.DataFrame) -> pd.DataFrame:
    columns = ["ticker", "period_end", "filed", "val"]
    if points.empty:
        return _empty(columns)
    ordered = points.sort_values(
        ["ticker", "period_end", "filed", "prefer"],
        kind="mergesort",
    ).drop_duplicates(["ticker", "period_end", "filed"], keep="last")
    return ordered.loc[:, columns]


def _flow(
    quarterly: pd.DataFrame,
    annual: pd.DataFrame,
    latest_name: str,
    ttm_name: str,
) -> pd.DataFrame:
    columns = ["ticker", "filed", latest_name, ttm_name]
    standalone = quarterly.loc[:, ["ticker", "period_end", "filed", "val"]].copy()
    standalone["prefer"] = 1
    pieces = [
        piece
        for piece in (_derive_q4(quarterly, annual), standalone)
        if not piece.empty
    ]
    if not pieces:
        return _empty(columns)
    points = _one_val(pd.concat(pieces, ignore_index=True))
    known = _known(points)
    if known.empty:
        return _empty(columns)
    known = known.sort_values(
        ["ticker", "filed", "period_end"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    known["rn"] = known.groupby(["ticker", "filed"], sort=False).cumcount() + 1
    latest = known.loc[known["rn"] == 1, ["ticker", "filed", "val"]].rename(
        columns={"val": latest_name}
    )
    window = known.loc[known["rn"] <= 4]
    ttm = window.groupby(["ticker", "filed"], as_index=False, sort=False).agg(
        n=("val", "size"),
        total=("val", "sum"),
    )
    ttm.loc[ttm["n"] < 4, "total"] = np.nan
    ttm = ttm.rename(columns={"total": ttm_name})
    return latest.merge(
        ttm.loc[:, ["ticker", "filed", ttm_name]],
        on=["ticker", "filed"],
        how="left",
    )


def _assemble(parts: list[pd.DataFrame], asset_keys: pd.DataFrame) -> pd.DataFrame:
    frames = [part for part in parts if not part.empty]
    if not frames:
        return _empty(PANEL_COLUMNS)
    wide = frames[0]
    for part in frames[1:]:
        wide = wide.merge(part, on=["ticker", "filed"], how="outer")
    wide = wide.sort_values(["ticker", "filed"], kind="mergesort")
    for column in [*LEVEL_COLUMNS, "asset_growth"]:
        if column not in wide.columns:
            wide[column] = np.nan
        wide[column] = pd.to_numeric(wide[column])
    wide[LEVEL_COLUMNS] = wide.groupby("ticker", sort=False)[LEVEL_COLUMNS].ffill()
    flags = (
        asset_keys.loc[:, ["ticker", "filed"]].drop_duplicates().assign(is_event=True)
    )
    wide = wide.merge(flags, on=["ticker", "filed"], how="left")
    wide["is_event"] = wide["is_event"].eq(True)
    wide["bucket"] = wide.groupby("ticker", sort=False)["is_event"].cumsum()
    wide["asset_growth"] = wide.groupby(["ticker", "bucket"], sort=False)[
        "asset_growth"
    ].ffill()
    wide = wide.loc[:, PANEL_COLUMNS]
    return wide.reset_index(drop=True)


def filing_panel(
    instant: pd.DataFrame,
    period: pd.DataFrame,
    members: pd.DataFrame,
) -> pd.DataFrame:
    """把公司事实收成每个披露日一行的面板

    时点科目取该披露日及之前已经出现过的记录里, period_end 最晚的一条
    之后只修订更早截止日时, 不把当前数退回比较期
    同一截止日有多个科目时, 元组里靠前的科目优先
    权益依次为 StockholdersEquity, 含少数股东权益的科目
    收入依次为 Revenues, RevenueFromContractWithCustomerExcludingAssessedTax,
    SalesRevenueNet
    净利润依次为 NetIncomeLoss, ProfitLoss
    股数是 dei 的 EntityCommonStockSharesOutstanding
    单季跨度包含 80 到 100 天, 全年跨度包含 350 到 380 天
    全年截止日还没有单季时, 第四季等于全年减去该财年内的前三季
    单季和推算的第四季撞在同一披露日时, 用单季
    net_income_ttm 是当时已经披露的最近四个单季之和, 凑不齐四个为缺失
    asset_growth 是当前资产除以大约一年前的资产再减 1
    一年前指间隔 300 到 430 天, 并且最接近 365 天的那份资产
    同一 CIK 有多只证券时, 每只证券各有一行
    不修改输入

    Args:
        instant: 时点事实, 列 cik, taxonomy, concept, unit, period_end, filed, val, accn
        period: 期间事实, 另有 period_start
        members: 列 ticker, cik, 一个 CIK 可以对应多只证券

    Returns:
        列 ticker, filed, assets, equity, shares, revenue_q, net_income_q,
        net_income_ttm, asset_growth
        每个 (ticker, filed) 一行, filed 升序
        某一列在该披露日没有新数时, 沿用该证券更早的披露
        资产披露日如果算不出同比增长, asset_growth 为缺失, 并且不沿用更早的增长率
        没有任何事实时返回带列名的空表

    Raises:
        KeyError: 输入缺少上面列出的列

    Example:
        >>> members = pd.DataFrame({"ticker": ["A"], "cik": ["1"]})
        >>> instant = pd.DataFrame([
        ...     {
        ...         "cik": "1", "taxonomy": "us-gaap", "concept": "Assets",
        ...         "unit": "USD", "period_end": "2024-12-31",
        ...         "filed": "2025-02-01", "val": 10.0, "accn": "a",
        ...     },
        ...     {
        ...         "cik": "1", "taxonomy": "us-gaap", "concept": "Assets",
        ...         "unit": "USD", "period_end": "2023-12-31",
        ...         "filed": "2025-02-01", "val": 8.0, "accn": "a",
        ...     },
        ... ])
        >>> period = pd.DataFrame(columns=[
        ...     "cik", "taxonomy", "concept", "unit", "period_start",
        ...     "period_end", "filed", "val", "accn",
        ... ])
        >>> panel = filing_panel(instant, period, members)
        >>> float(panel["assets"].iloc[0])
        10.0
        >>> float(panel["asset_growth"].iloc[0])
        0.25
    """

    assets = _asset_series(
        _instant_points(instant, members, ("Assets",), "us-gaap", "USD")
    )
    equity = _current(
        _instant_points(instant, members, EQUITY_CONCEPTS, "us-gaap", "USD"),
        "equity",
    )
    shares = _current(
        _instant_points(
            instant,
            members,
            (SHARES_CONCEPT,),
            "dei",
            "shares",
        ),
        "shares",
    )
    revenue_q = _period_points(period, members, REVENUE_CONCEPTS, QUARTER_DAYS)
    revenue_y = _period_points(period, members, REVENUE_CONCEPTS, ANNUAL_DAYS)
    income_q = _period_points(period, members, INCOME_CONCEPTS, QUARTER_DAYS)
    income_y = _period_points(period, members, INCOME_CONCEPTS, ANNUAL_DAYS)
    revenue = _flow(revenue_q, revenue_y, "revenue_q", "revenue_ttm").drop(
        columns=["revenue_ttm"]
    )
    income = _flow(income_q, income_y, "net_income_q", "net_income_ttm")
    return _assemble([assets, equity, shares, revenue, income], assets)


def read_filing_panel(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """从已入库的事实表读取申报日面板

    Args:
        connection: 可读的 DuckDB 连接, 含 universe, instant_facts, period_facts
            连接的关闭由调用方负责

    Returns:
        filing_panel 的结果
        只读取资产, 权益, 股数, 收入和净利润用到的科目

    Raises:
        duckdb.Error: 表不存在或查询失败

    Example:
        >>> import duckdb
        >>> connection = duckdb.connect()
        >>> _ = connection.execute(
        ...     "CREATE TABLE universe(ticker VARCHAR, cik VARCHAR)"
        ... )
        >>> _ = connection.execute("INSERT INTO universe VALUES ('A', '1')")
        >>> _ = connection.execute(
        ...     '''CREATE TABLE instant_facts(
        ...         cik VARCHAR, taxonomy VARCHAR, concept VARCHAR, unit VARCHAR,
        ...         period_end DATE, filed DATE, val DOUBLE, accn VARCHAR)'''
        ... )
        >>> _ = connection.execute(
        ...     '''CREATE TABLE period_facts(
        ...         cik VARCHAR, taxonomy VARCHAR, concept VARCHAR, unit VARCHAR,
        ...         period_start DATE, period_end DATE, filed DATE, val DOUBLE,
        ...         accn VARCHAR)'''
        ... )
        >>> _ = connection.execute(
        ...     "INSERT INTO instant_facts VALUES "
        ...     "('1', 'us-gaap', 'Assets', 'USD', '2024-12-31', '2025-02-01', 10, 'a')"
        ... )
        >>> float(read_filing_panel(connection)["assets"].iloc[0])
        10.0
    """

    instant_concepts = _sql_in(("Assets", *EQUITY_CONCEPTS))
    period_concepts = _sql_in((*REVENUE_CONCEPTS, *INCOME_CONCEPTS))
    instant = connection.execute(
        f"""
        SELECT cik, taxonomy, concept, unit, period_end, filed, val, accn
        FROM instant_facts
        WHERE (
            taxonomy = 'us-gaap' AND unit = 'USD'
            AND concept IN ({instant_concepts})
        ) OR (
            taxonomy = 'dei' AND unit = 'shares'
            AND concept = '{SHARES_CONCEPT}'
        )
        """
    ).df()
    period = connection.execute(
        f"""
        SELECT
            cik, taxonomy, concept, unit, period_start, period_end, filed, val, accn
        FROM period_facts
        WHERE taxonomy = 'us-gaap' AND unit = 'USD'
          AND concept IN ({period_concepts})
        """
    ).df()
    members = connection.execute("SELECT ticker, cik FROM universe").df()
    return filing_panel(instant, period, members)


def align_filings(prices: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """把申报日面板按 filed 对齐到行情的每个交易日

    每个交易日使用 filed 不晚于当天的最后一行
    披露当天可用
    不修改输入

    Args:
        prices: [date, ticker] MultiIndex DataFrame
        filings: 列至少含 ticker, filed, 其余列原样带上
            通常来自 filing_panel

    Returns:
        按 [date, ticker] 升序的 DataFrame
        保留行情列, 并带上 filings 里除 ticker 以外的列
        该证券在当天之前没有披露时, 这些列为缺失
        空的 filings 仍返回行情索引, 申报列为缺失

    Raises:
        KeyError: prices 没有 date 或 ticker 层, 或 filings 没有 filed

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"close": [5.0, 10.0]}, index=index)
        >>> filings = pd.DataFrame({
        ...     "ticker": ["A"],
        ...     "filed": pd.to_datetime(["2024-01-02"]),
        ...     "equity": [50.0],
        ... })
        >>> align_filings(prices, filings)["equity"].tolist()
        [50.0, 50.0]
    """

    left = prices.reset_index()
    left["date"] = pd.to_datetime(left["date"])
    if filings.empty:
        aligned = left.set_index(["date", "ticker"]).sort_index()
        for column in filings.columns:
            if column not in aligned.columns and column != "ticker":
                aligned[column] = np.nan
        return aligned
    right = filings.copy()
    right["date"] = pd.to_datetime(right["filed"])
    merged = pd.merge_asof(
        left.sort_values("date", kind="mergesort"),
        right.sort_values("date", kind="mergesort"),
        on="date",
        by="ticker",
        direction="backward",
    )
    return merged.set_index(["date", "ticker"]).sort_index()
