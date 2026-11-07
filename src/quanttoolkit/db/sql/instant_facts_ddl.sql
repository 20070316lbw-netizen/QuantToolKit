CREATE TABLE IF NOT EXISTS instant_facts(
    cik VARCHAR NOT NULL,
    taxonomy VARCHAR NOT NULL,
    concept VARCHAR NOT NULL,
    unit VARCHAR NOT NULL,
    period_end DATE NOT NULL,
    val DOUBLE NOT NULL,
    accn VARCHAR NOT NULL,
    form VARCHAR NOT NULL,
    filed DATE NOT NULL,
    fy INTEGER,
    fp VARCHAR,
    frame VARCHAR,                   -- 如 CY2025Q4I, 末尾 I 表示时点
    PRIMARY KEY (cik, taxonomy, concept, unit, period_end, accn)
)
