"""QuantToolKit 的公开数据接口目录, 不导入业务模块"""

_PANEL_SOURCES = (
    "panel.transfer.to_date_ticker_frame",
    "panel.transfer.prepare_price_panel",
    "sources.yahoo.get_prices",
    "panel.prices.fill_price_gaps",
)
_FACTOR_SOURCES = (
    "factors.mom_base.base_momfactor",
    "factors.reversal_base.base_reversalfactor",
    "factors.logvol_base.base_logvolfactor",
    "factors.rsi_base.base_rsifactor",
    "factors.exposure.cma_exposure",
)
_SCORES = (
    *_FACTOR_SOURCES,
    "factors.combine.combine_scores",
    "model.lambdarank.predict_scores",
    "panel.returns.historical_return",
    "panel.returns.log_returns",
    "panel.volatility.history_vol",
)
_WEIGHTS = (
    "portfolio.sizing.simple_quantile_long_short",
    "portfolio.sizing.vol_quantile_long_short",
)
_IC = ("metrics.ic", "metrics.rank_ic")
_BACKTESTS = ("vectorized", "event_driven")
_RESEARCH = ("factor_research",)
_RETURNS = (
    "metrics.prepare_portfolio_returns",
    "portfolio.benchmark.equal_weight_hold",
)
_BOOSTERS = ("model.lightgbm.fit_lightgbm", "model.lambdarank.fit_lambdarank")


def _input(requirements, *upstream, check=None, source="described"):
    return {
        "requirements": requirements,
        "upstream": upstream,
        "check": check,
        "source_policy": source,
    }


def _step(
    stage,
    definition,
    output,
    /,
    *,
    workflows=(),
    track=False,
    capture=None,
    inherit=None,
    **inputs,
):
    return {
        "stage": stage,
        "workflows": workflows,
        "definition": definition,
        "inputs": inputs,
        "output": output,
        "track": track,
        "capture": capture or {},
        "inherit": inherit,
    }


_price = _input(
    "已准备的 [date, ticker] DataFrame, price_col 必须是数值列, 非缺失价格有限且大于 0",
    *_PANEL_SOURCES,
    check="price",
    source="required",
)
_factor_price = _input(
    "[date, ticker] DataFrame, price_col 为数值列, 非缺失价格为正, 价格复权口径一致",
    *_PANEL_SOURCES,
    check="price",
    source="if_known",
)
_price_column = _input("价格列名, 默认值按对应函数签名")
_factor_frame = _input(
    "[date, ticker] 数值 DataFrame, 一列一个因子, 可用 pandas.concat 合并",
    *_FACTOR_SOURCES,
    "factors.neutral.neutralize_by_industry",
    "factors.zscore.zscore_by_date",
)
_score = _input(
    "[date, ticker] 数值 Series, 按日期做截面比较, 允许 NaN",
    *_SCORES,
    source="if_known",
)
_ic_score = {**_score, "check": "panel_numeric"}
_forward_return = _input(
    "对齐的 [date, ticker] 未来简单收益 Series, 可含 NaN",
    "panel.returns.future_returns",
    check="panel_numeric",
    source="if_known",
)
_calendar = _input(
    "共同交易日历 DatetimeIndex, 从完整行情的 date 层取值", *_PANEL_SOURCES
)
_feature = _input(
    "[date, ticker] 数值 DataFrame, 特征在基准日可得, 训练和预测列一致",
    "model.split.split_samples",
    "model.dataset.assemble_samples",
)
_label = _input(
    "[date, ticker] 数值 Series, 按索引对齐, 组装样本时必须有列名",
    "panel.returns.future_returns",
    "panel.volatility.forward_volatility",
    "model.split.split_samples",
)
_ic = _input("唯一日期索引 Series, 值为截面相关系数, 可含 NaN", *_IC, source="if_known")
_portfolio_returns = _input(
    "可排序的连续非重叠组合单期简单收益 Series, 值为小数, 缺失时指标为 NaN",
    *_RETURNS,
    source="if_known",
)
_plot_returns = _input(
    "唯一 DatetimeIndex, 非空有限数值 Series, 值为连续非重叠单期组合简单收益",
    *_RETURNS,
    source="required",
)
_axes = _input("可选 Matplotlib Axes, 默认新建")
_X = _input("二维数值特征数组或列表")
_y = _input("与特征行一一对应的一维数值标签")
_horizon = _input("正整数, 按各证券记录计数")
_gap = _input("非负整数, 基准记录到入场的间隔")

CATALOG = {
    "sources.yahoo.get_prices": _step(
        "数据接入",
        "从 Yahoo 下载并整理指定周期的行情",
        "排序后的行情面板 DataFrame",
        track=True,
        tickers=_input("证券代码序列"),
        start=_input("起始日期"),
        end=_input("结束日期或 None"),
        auto_adjust=_input("是否取复权行情"),
        interval=_input("行情周期, 默认日频 1d"),
        progress=_input("是否显示进度"),
    ),
    "sources.companyfacts.download_company_facts": _step(
        "数据接入",
        "下载 SEC EDGAR 的 Company Facts 批量归档",
        "通过 ZIP 目录检查的本地归档路径",
        destination=_input("本地 ZIP 路径, 父目录须存在"),
        user_agent=_input("含调用者名称和联系邮箱的 SEC User-Agent"),
    ),
    "sources.companyfacts.read_company_facts": _step(
        "数据接入",
        "按 CIK 读取 SEC XBRL 事实, 分为期间和时点",
        "(period_facts, instant_facts), 保留单位, 申报日和 accession",
        zip_path=_input(
            "本地 Company Facts ZIP", "sources.companyfacts.download_company_facts"
        ),
        ciks=_input("十位 CIK 字符串序列"),
    ),
    "db.companyfacts.write_company_facts": _step(
        "数据接入",
        "在同一 DuckDB 事务写入期间和时点事实",
        "(期间写入行数, 时点写入行数)",
        connection=_input("可写 DuckDB 连接", "db.connection.get_duckdb"),
        period=_input("期间事实表", "sources.companyfacts.read_company_facts"),
        instant=_input("时点事实表", "sources.companyfacts.read_company_facts"),
    ),
    "fundamentals.fundamental_fact_mapping": _step(
        "数据接入",
        "查询已登记的 SEC 标签与规范财务指标",
        "普通列 DataFrame, 含 fact_kind, taxonomy, concept, metric, priority",
        workflows=_RESEARCH,
    ),
    "fundamentals.read_fundamental_facts": _step(
        "数据接入",
        "读取指定 CIK 的规范财务事实, 保留报告单位和申报编号",
        "含 metric, 原始 taxonomy, concept, unit, filed, accn 的普通列 DataFrame",
        connection=_input(
            "含期间和时点事实表的 DuckDB 连接", "db.connection.get_duckdb"
        ),
        ciks=_input("十位 CIK 字符串序列"),
        workflows=_RESEARCH,
    ),
    "fundamentals.filing_panel": _step(
        "数据接入",
        "从公司事实整理每只证券在各申报日的财务指标",
        "按 ticker 和 filed 排序的申报日面板",
        instant=_input("时点事实表", "sources.companyfacts.read_company_facts"),
        period=_input("期间事实表", "sources.companyfacts.read_company_facts"),
        members=_input("含 ticker, cik 的证券映射"),
        workflows=_RESEARCH,
    ),
    "fundamentals.read_filing_panel": _step(
        "数据接入",
        "从已入库的事实表读取申报日面板",
        "按 ticker 和 filed 排序的申报日面板",
        connection=_input(
            "含期间和时点事实表的 DuckDB 连接", "db.connection.get_duckdb"
        ),
        members=_input("含 ticker, cik 的证券映射"),
        workflows=_RESEARCH,
    ),
    "fundamentals.align_filings": _step(
        "数据接入",
        "按证券将申报日面板接到日线, 使用严格早于交易日的申报",
        "按 [date, ticker] 排序的行情及财务列",
        prices=_input("[date, ticker] 行情面板", *_PANEL_SOURCES),
        filings=_input(
            "含 ticker, filed 的面板",
            "fundamentals.filing_panel",
            "fundamentals.read_filing_panel",
        ),
        workflows=_RESEARCH,
    ),
    "sources.french.read_french_factors": _step(
        "数据接入",
        "解析 French 日频因子",
        "date 索引 DataFrame, 收益为小数",
        path=_input("French 日频 CSV 或 ZIP 的文件路径"),
    ),
    "sources.wikipedia.fetch_sp500_universe": _step(
        "数据接入",
        "下载当前标普 500 成员",
        "静态成员 DataFrame, 含 ticker、name、sub_industry",
    ),
    "db.connection.get_duckdb": _step(
        "数据接入",
        "打开 DuckDB 连接",
        "DuckDBPyConnection",
        path=_input("数据库文件路径", "data_pipeline.main"),
        read_only=_input("是否只读, 默认 True"),
    ),
    "db.operation.read_sql": _step(
        "数据接入",
        "读取 SQL 文件",
        "UTF-8 SQL 字符串",
        sql=_input("SQL 文件路径"),
    ),
    "db.operation.operation_sql": _step(
        "数据接入",
        "在同一事务建表与写入",
        "最后一条语句返回的整数计数",
        sql=_input("写入 SQL", "db.operation.read_sql"),
        ddl=_input("建表 SQL", "db.operation.read_sql"),
        connection=_input("可写 DuckDB 连接", "db.connection.get_duckdb"),
        df=_input("与写入 SQL 对应的普通列 DataFrame, 行情入库前还原索引"),
    ),
    "data_pipeline.main": _step(
        "数据接入",
        "准备名单, 行情, French 因子和 SEC 公司事实, 再建立 DuckDB 视图",
        "数据库文件绝对路径",
        logger_path=_input("可选日志目录"),
        data_path=_input("可选数据目录"),
        sec_user_agent=_input(
            "SEC User-Agent, 含调用者名称和联系邮箱, 默认复用本地文件"
        ),
    ),
    "panel.transfer.to_date_ticker_frame": _step(
        "行情准备",
        "将普通列的 date 和 ticker 转为排序面板索引",
        "[date, ticker] DataFrame",
        track=True,
        df=_input("含可转换的 date 列和证券代码 ticker 列的长表 DataFrame"),
    ),
    "panel.transfer.prepare_price_panel": _step(
        "行情准备",
        "校验并接入已经有面板索引的外部行情",
        "保留输入顺序的行情面板 DataFrame",
        track=True,
        prices=_input(
            "[date, ticker] DataFrame, 键唯一",
            *_PANEL_SOURCES,
            check="panel_frame",
            source="if_known",
        ),
    ),
    "panel.prices.fill_price_gaps": _step(
        "行情准备",
        "补齐证券首末记录之间的共同交易日",
        "排序面板, 新行价格沿用前值、成交量为 0",
        track=True,
        prices=_input(
            "[date, ticker] DataFrame, 键唯一",
            *_PANEL_SOURCES,
            check="panel_frame",
            source="if_known",
        ),
    ),
    "panel.returns.future_returns": _step(
        "标签计算",
        "用入场和出场价格计算未来持有期简单收益",
        "[date, ticker] 未来收益标签 Series",
        track=True,
        capture={"n_periods": "n_periods", "gap": "gap"},
        df=_price,
        n_periods=_horizon,
        gap=_gap,
        price_col=_price_column,
    ),
    "panel.returns.historical_return": _step(
        "特征计算",
        "用当前价格和历史价格计算回看期简单收益",
        "[date, ticker] 历史收益 Series",
        track=True,
        df=_price,
        n_periods=_horizon,
        price_col=_price_column,
    ),
    "panel.returns.log_returns": _step(
        "特征计算",
        "按证券计算相邻记录的对数收益",
        "[date, ticker] 单期对数收益 Series",
        track=True,
        df=_price,
        price_col=_price_column,
    ),
    "panel.volatility.history_vol": _step(
        "特征计算",
        "计算截至当期的滚动样本波动率",
        "[date, ticker] 历史波动率 Series, 不年化",
        track=True,
        df=_price,
        window=_horizon,
        price_col=_price_column,
    ),
    "panel.volatility.forward_volatility": _step(
        "标签计算",
        "计算未来持有区间的滚动样本波动率",
        "[date, ticker] 未来波动率标签 Series",
        track=True,
        capture={"n_periods": "window", "gap": "gap"},
        df=_price,
        window=_horizon,
        gap=_gap,
        price_col=_price_column,
    ),
    "factors.mom_base.base_momfactor": _step(
        "特征计算",
        "计算回看窗口的动量",
        "[date, ticker] 动量 Series",
        track=True,
        prices=_factor_price,
        window=_horizon,
        price_col=_price_column,
        skip=_input("跳过最近的记录数, 默认 0"),
    ),
    "factors.reversal_base.base_reversalfactor": _step(
        "特征计算",
        "计算窗口内的短期反转因子",
        "[date, ticker] 反转 Series",
        track=True,
        prices=_factor_price,
        window=_horizon,
        price_col=_price_column,
        skip=_input("跳过最近的记录数, 默认 0"),
    ),
    "factors.logvol_base.base_logvolfactor": _step(
        "特征计算",
        "计算滚动对数收益样本波动率",
        "[date, ticker] 对数波动率 Series",
        track=True,
        prices=_factor_price,
        window=_horizon,
        price_col=_price_column,
    ),
    "factors.rsi_base.base_rsifactor": _step(
        "特征计算",
        "按 Wilder 方法计算 RSI",
        "[date, ticker] RSI Series, 取值 0 到 100",
        track=True,
        prices=_factor_price,
        window=_horizon,
        price_col=_price_column,
    ),
    "factors.exposure.cma_exposure": _step(
        "特征计算",
        "用滚动 FF5 回归估计 CMA 暴露",
        "[date, ticker] CMA 系数 Series",
        track=True,
        prices=_input("含 adj_close 和 volume 的排序行情面板", *_PANEL_SOURCES),
        ff=_input(
            "日频因子 DataFrame, 含 RF 和 FF5 小数收益列",
            "sources.french.read_french_factors",
        ),
        window=_horizon,
    ),
    "factors.neutral.neutralize_by_industry": _step(
        "因子加工",
        "逐日减去各行业截面均值",
        "同行同列的行业中性因子 DataFrame",
        factors=_factor_frame,
        industry=_input(
            "ticker 索引的行业标签 Series, 可由成员表提取",
            "sources.wikipedia.fetch_sp500_universe",
        ),
        date_level=_input("日期层名称"),
        ticker_level=_input("证券层名称"),
    ),
    "factors.zscore.zscore_by_date": _step(
        "因子加工",
        "逐日对各因子做截面标准化",
        "同行同列的标准分数 DataFrame",
        factors=_factor_frame,
        date_level=_input("日期层名称"),
    ),
    "factors.combine.combine_scores": _step(
        "信号构造",
        "对有效因子列加权合成截面分数",
        "[date, ticker] score Series",
        track=True,
        z=_factor_frame,
        weights=_input("因子列名到权重的可选映射"),
    ),
    "portfolio.rebalance.rebalance_dates": _step(
        "策略构造",
        "从共同日历每隔指定期数取一个调仓日",
        "调仓日 DatetimeIndex",
        dates=_calendar,
        freq=_input("调仓间隔交易日数"),
    ),
    "portfolio.rebalance.rebalance_dates_belong": _step(
        "策略构造",
        "将日历日期映射到最近的调仓日",
        "日期到调仓日的 Series",
        dates=_calendar,
        freq=_input("调仓间隔交易日数"),
    ),
    "portfolio.sizing.simple_quantile_long_short": _step(
        "策略构造",
        "每个截面选最高和最低分位做等权多空",
        "[date, ticker] 目标权重 Series",
        track=True,
        score=_score,
        n_quantiles=_input("至少 2 组"),
        date_level=_input("日期层名称"),
    ),
    "portfolio.sizing.vol_quantile_long_short": _step(
        "策略构造",
        "先按风险指标分层再对分数做分位多空",
        "[date, ticker] 目标权重 Series",
        track=True,
        score=_score,
        vol=_input(
            "与分数对齐的风险指标 Series",
            "panel.volatility.history_vol",
            "factors.logvol_base.base_logvolfactor",
        ),
        n_vol_groups=_input("波动率分组数"),
        n_quantiles=_input("每层内分位数, 至少 2"),
        date_level=_input("日期层名称"),
    ),
    "portfolio.benchmark.equal_weight_hold": _step(
        "组合绩效",
        "首日等金额买入后持有份额不变",
        "日期索引的基准单期简单收益, 首期 NaN",
        workflows=_BACKTESTS,
        track=True,
        prices=_factor_price,
        price_col=_price_column,
    ),
    "model.dataset.label_end_dates": _step(
        "标签计算",
        "找出每个未来收益标签使用的最后一条价格日期",
        "[date, ticker] 日期 Series, 不足为 NaT",
        track=True,
        capture={"n_periods": "n_periods", "gap": "gap"},
        index=_input(
            "完整行情的 [date, ticker] MultiIndex", *_PANEL_SOURCES, check="panel_index"
        ),
        n_periods=_horizon,
        gap=_gap,
    ),
    "model.dataset.prepare_label_end_dates": _step(
        "标签计算",
        "接入外部提供的实际标签结束日期",
        "[date, ticker] 日期 Series",
        track=True,
        inherit="values",
        values=_input(
            "[date, ticker] Series, 值为日期, 可含 NaT",
            "model.dataset.label_end_dates",
            "model.dataset.prepare_label_end_dates",
            check="label_end",
            source="if_known",
        ),
    ),
    "model.dataset.assemble_samples": _step(
        "样本准备",
        "按索引对齐特征和标签并保留完整行",
        "[date, ticker] DataFrame, 标签列在最后",
        track=True,
        inherit="label",
        features=_feature,
        label=_label,
    ),
    "model.split.split_samples": _step(
        "样本准备",
        "按共同日历切分并排除实际标签跨段的行",
        "train、valid、test 到样本表的字典",
        samples=_input(
            "[date, ticker] 样本 DataFrame, 允许从外部或数据库读取",
            "model.dataset.assemble_samples",
            "panel.transfer.to_date_ticker_frame",
            check="panel_frame",
            source="if_known",
        ),
        dates=_calendar,
        label_end=_input(
            "逐行标签结束日期 Series, 已知时须匹配标签期限",
            "model.dataset.label_end_dates",
            "model.dataset.prepare_label_end_dates",
            check="label_end",
            source="required",
        ),
        train_end=_input("训练截止日, 含当天"),
        valid_end=_input("验证截止日, 含当天"),
        embargo=_input("训练和验证末尾隔离交易日数"),
    ),
    "model.split.split_dates": _step(
        "样本准备",
        "给共同日历标记训练、验证、测试和隔离期",
        "日期索引 Series, 隔离期为缺失",
        dates=_calendar,
        train_end=_input("训练截止日"),
        valid_end=_input("验证截止日"),
        embargo=_input("隔离交易日数, 非负"),
    ),
    "model.relevance.relevance_by_date": _step(
        "样本准备",
        "将截面标签按相对秩离散成相关度",
        "[date, ticker] 整数等级 Series",
        label=_label,
        n_grades=_input("整数档位数, 默认 31"),
    ),
    "model.lambdarank.group_sizes": _step(
        "模型训练",
        "统计连续日期分组的行数",
        "每组一个元素的整数 ndarray",
        index=_input(
            "按日期成组排列的 [date, ticker] MultiIndex", "model.split.split_samples"
        ),
    ),
    "model.lambdarank.fit_lambdarank": _step(
        "模型训练",
        "用同日证券分组训练 LightGBM 排序模型",
        "训练后的 lightgbm.Booster",
        features=_feature,
        relevance=_input(
            "对齐的 0 到 30 整数 Series", "model.relevance.relevance_by_date"
        ),
        params=_input("可选 LambdaRankParams"),
    ),
    "model.lightgbm.fit_lightgbm": _step(
        "模型训练",
        "训练 LightGBM 回归, 可用验证集早停",
        "训练后的 lightgbm.Booster",
        features=_feature,
        label=_label,
        valid_features=_feature,
        valid_label=_label,
        params=_input("可选 LightGBMParams, 验证特征和标签成对提供"),
    ),
    "model.tree.RegressionTree.fit": _step(
        "模型训练",
        "用平方误差拟合回归树",
        "已拟合的 RegressionTree 自身",
        X=_X,
        y=_y,
    ),
    "model.tree.RegressionTree.predict": _step(
        "模型预测",
        "沿回归树找到每行的叶子值",
        "逐行预测值 ndarray",
        self=_input("已训练的树", "model.tree.RegressionTree.fit"),
        X=_X,
    ),
    "model.tree.RegressionTree.apply": _step(
        "模型预测",
        "沿回归树找到每行的叶子编号",
        "逐行叶子编号 ndarray",
        self=_input("已训练的树", "model.tree.RegressionTree.fit"),
        X=_X,
    ),
    "model.gbdt.SimpleGBDT.fit": _step(
        "模型训练",
        "用残差迭代训练回归树, 可用验证集早停",
        "已拟合的 SimpleGBDT 自身",
        X=_X,
        y=_y,
        X_val=_X,
        y_val=_y,
        early_stopping_rounds=_input("可选早停轮数"),
    ),
    "model.gbdt.SimpleGBDT.predict": _step(
        "模型预测",
        "叠加已训练各棵树的预测",
        "逐行预测值 ndarray",
        self=_input("已训练的 GBDT", "model.gbdt.SimpleGBDT.fit"),
        X=_X,
    ),
    "model.lambdarank.predict_scores": _step(
        "模型预测",
        "按特征表行顺序预测回归值或排序分数",
        "[date, ticker] pred Series",
        track=True,
        booster=_input("已训练或从文件恢复的 LightGBM Booster", *_BOOSTERS),
        features=_feature,
    ),
    "metrics.ic": _step(
        "因子研究",
        "逐日计算分数与未来收益的 Pearson 相关",
        "日期索引 IC Series",
        workflows=_RESEARCH,
        track=True,
        factor=_ic_score,
        forward_return=_forward_return,
    ),
    "metrics.rank_ic": _step(
        "因子研究",
        "逐日计算分数与未来收益的 Spearman 相关",
        "日期索引 Rank IC Series",
        workflows=_RESEARCH,
        track=True,
        factor=_ic_score,
        forward_return=_forward_return,
    ),
    "metrics.ic_summary": _step(
        "因子研究",
        "汇总 IC 均值、标准差、ICIR 和正值比例",
        "四项统计量组成的 Series",
        workflows=_RESEARCH,
        ic=_ic,
    ),
    "metrics.prepare_portfolio_returns": _step(
        "组合绩效",
        "接入调用方提供的组合单期简单收益",
        "日期索引组合单期简单收益 Series",
        workflows=_BACKTESTS,
        track=True,
        returns=_input(
            "唯一日期索引的非空有限数值 Series, 由调用方确认连续且非重叠",
            *_RETURNS,
            check="dated_finite",
            source="if_known",
        ),
    ),
    "metrics.prepare_forward_spread": _step(
        "因子研究",
        "接入逐信号日的远期多空收益差",
        "日期索引的远期多空价差 Series",
        workflows=_RESEARCH,
        track=True,
        spread=_input(
            "唯一日期索引的数值 Series, 允许 NaN, 持有窗口可以重叠",
            "metrics.prepare_forward_spread",
            check="dated_numeric",
            source="if_known",
        ),
    ),
    "metrics.total_return": _step(
        "组合绩效",
        "从 1 复利得到区间总收益",
        "小数收益 float",
        workflows=_BACKTESTS,
        returns=_portfolio_returns,
    ),
    "metrics.sharpe": _step(
        "组合绩效",
        "用均值除以样本标准差",
        "未年化夏普 float",
        workflows=_BACKTESTS,
        returns=_portfolio_returns,
    ),
    "metrics.max_drawdown": _step(
        "组合绩效",
        "求组合净值相对历史峰值的最低比例",
        "非正的小数回撤 float",
        workflows=_BACKTESTS,
        returns=_portfolio_returns,
    ),
    "metrics.win_rate": _step(
        "收益评估",
        "统计观测收益严格为正的比例",
        "正收益比例 float",
        workflows=(*_BACKTESTS, *_RESEARCH),
        returns=_input(
            "一维数值收益 Series, 可用于组合单期收益或逐观察日的远期价差",
            *_RETURNS,
            "metrics.prepare_forward_spread",
            source="if_known",
        ),
    ),
    "figure.plots.plot_nav": _step(
        "报告",
        "绘制从 1 起算的组合复利净值",
        "Matplotlib Axes",
        workflows=_BACKTESTS,
        portfolio_returns=_plot_returns,
        ax=_axes,
    ),
    "figure.plots.plot_drawdown": _step(
        "报告",
        "绘制组合净值相对历史峰值的回撤",
        "Matplotlib Axes",
        workflows=_BACKTESTS,
        portfolio_returns=_plot_returns,
        ax=_axes,
    ),
    "figure.plots.plot_target_exposure": _step(
        "报告",
        "按调仓日绘制目标权重的总敞口和净敞口",
        "Matplotlib Axes",
        workflows=("vectorized",),
        target_weights=_input(
            "唯一 [date, ticker] 权重 Series, 小数权重、空头为负",
            *_WEIGHTS,
            source="if_known",
        ),
        ax=_axes,
    ),
    "figure.plots.plot_execution_slippage": _step(
        "报告",
        "以基点为单位绘制逐笔滑点",
        "Matplotlib Axes",
        workflows=("event_driven",),
        slippage_bps=_input(
            "成交时间索引的数值 Series, 允许重复时间, 正值表示不利成交",
            source="if_known",
        ),
        ax=_axes,
    ),
    "figure.plots.plot_ic": _step(
        "报告",
        "绘制逐日截面相关, 保留缺口",
        "Matplotlib Axes",
        workflows=_RESEARCH,
        ic=_ic,
        ax=_axes,
    ),
    "figure.plots.plot_forward_spread": _step(
        "报告",
        "按信号日绘制远期多空价差, 各观测值独立展示",
        "Matplotlib Axes",
        workflows=_RESEARCH,
        forward_spread=_input(
            "唯一日期索引的远期价差 Series, 值为小数, 可含 NaN",
            "metrics.prepare_forward_spread",
            source="if_known",
        ),
        ax=_axes,
    ),
    "figure.report.plot_report": _step(
        "报告",
        "根据研究或回测流程拼接共享时间轴的图表",
        "Matplotlib Figure, 可写出图文件",
        workflows=(*_BACKTESTS, *_RESEARCH),
        kind=_input("vectorized、event_driven 或 factor_research"),
        output=_input("可选输出路径, 父目录须存在"),
        artifacts=_input(
            "按报告类型提供必需和可选的命名产物, 由各单图契约检查",
            *_RETURNS,
            *_WEIGHTS,
            *_IC,
            "metrics.prepare_forward_spread",
        ),
    ),
    "model.lightgbm.LightGBMParams.booster_params": _step(
        "模型配置",
        "提取 LightGBM 回归 Booster 配置",
        "不含训练轮数和早停轮数的字典",
    ),
    "model.lambdarank.LambdaRankParams.booster_params": _step(
        "模型配置",
        "提取 LambdaRank 目标与随机种子",
        "LightGBM Booster 参数字典",
    ),
    "model.params.ModelParams.to_dict": _step(
        "模型配置",
        "将模型配置整理为可序列化字段",
        "独立的字段字典",
    ),
    "model.params.ModelParams.save": _step(
        "模型配置",
        "保存模型参数",
        "None, 在指定路径写入 JSON",
        path=_input("JSON 输出路径, 父目录须存在"),
    ),
    "model.params.ModelParams.load": _step(
        "模型配置",
        "从 JSON 恢复模型参数",
        "当前参数类的实例",
        path=_input("JSON 文件路径", "model.params.ModelParams.save"),
    ),
    "model.params.TrainLog.save": _step(
        "模型记录",
        "保存模型名、训练参数和指标",
        "None, 在指定路径写入 JSON",
        path=_input("JSON 输出路径, 父目录须存在"),
    ),
    "model.params.TrainLog.load": _step(
        "模型记录",
        "从 JSON 恢复训练记录",
        "TrainLog 实例",
        path=_input("JSON 文件路径", "model.params.TrainLog.save"),
    ),
    "tracelog.trace": _step(
        "追踪",
        "记录函数耗时、调用链和可选的返回值摘要",
        "装饰后的函数或装饰器",
        func=_input("可选的待装饰函数"),
        max_rows=_input("预览的行数或元素个数, 默认 5"),
        log_return=_input("是否记录正常返回值, 默认 True"),
    ),
    "tracelog.configure_log": _step(
        "追踪",
        "设置追踪日志路径",
        "设置后的 Path",
        path=_input("日志文件路径, 父目录可自动创建"),
    ),
    "tracelog.clear_log": _step(
        "追踪",
        "清空指定或当前追踪文件",
        "被清空的日志 Path",
        path=_input("可选日志路径, 默认使用当前设置", "tracelog.configure_log"),
    ),
    "tracelog.print_call_chain": _step(
        "追踪",
        "打印已记录的函数调用链",
        "调用链文本, 无记录时为空",
    ),
}
