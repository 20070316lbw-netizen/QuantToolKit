CREATE TABLE IF NOT EXISTS ff_factors(
    date DATE NOT NULL,
    model VARCHAR NOT NULL,    -- ff3 或 ff5, 两者的 SMB 不是同一条序列
    mkt_rf DOUBLE,
    smb DOUBLE,
    hml DOUBLE,
    rmw DOUBLE,                -- 只有 ff5 有值
    cma DOUBLE,                -- 只有 ff5 有值
    rf DOUBLE,
    PRIMARY KEY(model, date)
)
