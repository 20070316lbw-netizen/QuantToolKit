CREATE TABLE IF NOT EXISTS prices(
    date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE NOT NULL,
    adj_close DOUBLE,
    volume DOUBLE,
    PRIMARY KEY(ticker, date)
)
