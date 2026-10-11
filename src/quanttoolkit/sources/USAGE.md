## SEC 官方 companyfacts.zip 获取方法:

```bash
curl -H "User-Agent: 你的名字 你的邮箱" \
  -o data/companyfacts.zip \
  https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
```


## 官方映射表获取方法

```bash
curl -H "User-Agent: 你的名字 you@example.com" \
  -o data/company_tickers.json https://www.sec.gov/files/company_tickers.json
```

先下载映射文件, 再抓取最新标普500名单:

```python
from quanttoolkit.sources.wikipedia import fetch_sp500_universe

universe = fetch_sp500_universe(json_path="data/company_tickers.json")
```

返回普通 DataFrame, 列为 `ticker`, `name`, `sub_industry`, `cik`, `sec_title`, 按 ticker 升序排列
`cik` 为10位字符串, `sec_title` 使用 JSON 的 `cik_str` 对应 `title` 填充, 未匹配时为缺失值
映射文件必须事先下载, 文件不存在时抛出 `FileNotFoundError: 没有映射文件`, 并提示按上述方法获取

## 抓取到入库

先按上面的命令下载 `data/company_tickers.json`, 再从仓库根目录运行:

```bash
uv run python scripts/ingest_universe.py
```

脚本抓取最新成员并映射 SEC 名称, 写入 `data/quanttoolkit.db` 的 `universe` 表
首次运行自动建表, 再次运行完整替换成员名单, 写入过程使用事务, 失败时保留原数据
运行结束会显示抓取记录数, 入库记录数, SEC 名称缺失数和数据库路径

单独调用入库函数:

```python
from pathlib import Path

from quanttoolkit.db.connection import get_duckdb
from quanttoolkit.db.ingestion import ingestion_universe
from quanttoolkit.sources.wikipedia import fetch_sp500_universe

universe = fetch_sp500_universe(json_path="data/company_tickers.json")
with get_duckdb(path=Path("data/quanttoolkit.db"), read_only=False) as connection:
    written = ingestion_universe(df=universe, connection=connection)
```
