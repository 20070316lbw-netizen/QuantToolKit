WITH base_data AS (
    -- 1. 关联股票池, 计算基础单期涨跌与对数收益率
    SELECT 
        p.date,
        p.ticker,
        u.sub_industry,
        p.adj_close,
        p.adj_close - LAG(p.adj_close, 1) OVER w AS diff,
        LN(p.adj_close / LAG(p.adj_close, 1) OVER w) AS log_ret,
        LN(LEAD(p.adj_close, 5) OVER w / p.adj_close) AS fwd_ret_5d
    FROM prices p
    INNER JOIN (SELECT DISTINCT ticker, sub_industry FROM universe) u ON p.ticker = u.ticker
    WHERE p.adj_close > 0
    WINDOW w AS (PARTITION BY p.ticker ORDER BY p.date)
),
raw_factors AS (
    SELECT 
        date,
        ticker,
        sub_industry,
        fwd_ret_5d,
        
        CASE WHEN COUNT(log_ret) OVER w20 = 20 
             THEN STDDEV_SAMP(log_ret) OVER w20 
        END AS logvol_20,

        CASE WHEN COUNT(log_ret) OVER w20 = 20 
             THEN SUM(log_ret) OVER w20 
        END AS raw_mom_20,

        CASE WHEN COUNT(log_ret) OVER w5 = 5 
             THEN -SUM(log_ret) OVER w5 
        END AS reversal_5,

        CASE WHEN diff IS NOT NULL THEN GREATEST(diff, 0) END AS gain,
        CASE WHEN diff IS NOT NULL THEN GREATEST(-diff, 0) END AS loss
    FROM base_data
    WINDOW 
        w5  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
        w20 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
),
factor_table AS (
    SELECT 
        date,
        ticker,
        sub_industry,
        fwd_ret_5d,
        logvol_20,
        LAG(raw_mom_20, 1) OVER (PARTITION BY ticker ORDER BY date) AS mom_20_1,
        reversal_5,
        CASE WHEN COUNT(gain) OVER w14 = 14 THEN
            100.0 * AVG(gain) OVER w14 / NULLIF(AVG(gain) OVER w14 + AVG(loss) OVER w14, 0)
        END AS rsi_14
    FROM raw_factors
    WINDOW w14 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
)

-- 截面标准化和加权在 zscore_by_date, combine_scores
SELECT
    date,
    ticker,
    sub_industry,
    fwd_ret_5d,
    mom_20_1,
    reversal_5,
    rsi_14,
    logvol_20
FROM factor_table
