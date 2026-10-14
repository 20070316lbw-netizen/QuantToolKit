CREATE TABLE IF NOT EXISTS period_facts(
    cik VARCHAR NOT NULL,
    taxonomy VARCHAR NOT NULL,       -- us-gaap, dei, ifrs-full
    concept VARCHAR NOT NULL,
    unit VARCHAR NOT NULL,           -- USD, USD/shares, shares
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    val DOUBLE NOT NULL,
    accn VARCHAR NOT NULL,           -- 申报编号
    form VARCHAR NOT NULL,
    filed DATE NOT NULL,             -- 申报日, 这一天之后才能使用该数字
    fy INTEGER,                      -- 申报所属财年, 比较期沿用这份申报的 fy
    fp VARCHAR,                      -- 申报所属期间, FY Q1 Q2 Q3
    frame VARCHAR,                   -- SEC 日历框, 如 CY2025, CY2025Q3
    PRIMARY KEY (cik, taxonomy, concept, unit, period_start, period_end, accn)
)
