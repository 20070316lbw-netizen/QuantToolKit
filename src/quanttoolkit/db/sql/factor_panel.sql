-- 把价格因子和 FF5 放在同一个视图里
-- 从 prices 的第 64 个不同交易日开始保留记录
-- mom_63 需要 63 条对数收益, 也就是 64 条价格
-- 只保留这一天及之后, 并且当天要有 ff5
-- 补齐日的 volume 为 0, 这里把它转成缺失, 隔夜收益也留空
-- 动量和对数波动率仍使用补齐后的价格, 与 Python 因子的口径一致
CREATE OR REPLACE VIEW factor_panel AS
WITH px AS (
    SELECT
        date,
        ticker,
        open,
        close,
        adj_close,
        NULLIF(volume, 0) AS volume,
        ln(adj_close / lag(adj_close) OVER w) AS log_ret,
        CASE
            WHEN volume > 0 AND open > 0 AND lag(close) OVER w > 0
            THEN ln(open / lag(close) OVER w)
        END AS overnight,
        CASE
            WHEN volume > 0 AND open > 0 AND close > 0
            THEN ln(close / open)
        END AS intraday,
        CASE
            WHEN volume > 0 AND lag(volume) OVER w > 0
            THEN ln(volume / lag(volume) OVER w)
        END AS dvol
    FROM prices
    WINDOW w AS (PARTITION BY ticker ORDER BY date)
),
rolled AS (
    SELECT
        date,
        ticker,
        overnight,
        intraday,
        dvol,
        volume,
        CASE
            WHEN count(log_ret) OVER w21 = 21 THEN sum(log_ret) OVER w21
        END AS mom_21,
        CASE
            WHEN count(log_ret) OVER w63 = 63 THEN sum(log_ret) OVER w63
        END AS mom_63,
        CASE
            WHEN count(log_ret) OVER w21 = 21 THEN stddev_samp(log_ret) OVER w21
        END AS logvol_21,
        CASE
            WHEN count(volume) OVER w21 = 21
            THEN volume / avg(volume) OVER w21
        END AS rel_volume_21
    FROM px
    WINDOW
        w21 AS (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
        ),
        w63 AS (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 62 PRECEDING AND CURRENT ROW
        )
),
start_line AS (
    SELECT date AS start_date
    FROM (
        SELECT date, row_number() OVER (ORDER BY date) AS n
        FROM (SELECT DISTINCT date FROM prices)
    )
    WHERE n = 64
)
SELECT
    rolled.date,
    rolled.ticker,
    ff.mkt_rf,
    ff.smb,
    ff.hml,
    ff.rmw,
    ff.cma,
    ff.rf,
    rolled.mom_21,
    rolled.mom_63,
    rolled.logvol_21,
    rolled.overnight,
    rolled.intraday,
    rolled.dvol,
    rolled.rel_volume_21
FROM rolled
JOIN ff_factors AS ff
    ON ff.date = rolled.date
    AND ff.model = 'ff5'
WHERE rolled.date >= (SELECT start_date FROM start_line);
