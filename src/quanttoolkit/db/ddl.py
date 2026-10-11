"""DuckDB 数据库建表语句"""

UNIVERSE_DDL = """
    CREATE TABLE IF NOT EXISTS universe(
        ticker VARCHAR,
        name VARCHAR,
        sub_industry VARCHAR,       -- 行业分布
        cik VARCHAR,
        sec_title VARCHAR           -- SEC company_tickers.json 里面的公司名称
        )
    """
