# 一次准备好数据

装进其他项目后, 在自己的项目目录调用:

```python
from quanttoolkit.data_pipeline import main

database = main()
```

不用先手动下载 ZIP, 也不用另外拼建表和写入代码
返回值是数据库的绝对路径, 可以直接交给 `get_duckdb(path=database)`
导入模块不会开始下载, 调用 `main()` 才开始

也可以只改文件夹位置, 两个参数都支持字符串和 `Path`:

```python
from quanttoolkit.data_pipeline import main

database = main(logger_path="logs", data_path="data")
```

未传入的路径分别使用当前工作目录下的 `logs/` 和 `data/`
这里把运行时的工作目录作为项目根目录, 不自动向上寻找仓库
相对路径也是从工作目录开始算, `~` 会展开为用户目录
不会使用安装包所在的 `site-packages` 作为数据目录

## 跑了哪些步骤

1. 建好日志和数据文件夹, 输出两者的实际位置, 使用默认路径时额外说明
2. 从 Wikipedia 获取当前标普500成员, 写入 `universe`
3. 下载官方 FF3 和 FF5 日频 ZIP, 放在数据目录, 解析后一起写入 `ff_factors`
4. 按本次名单下载 2016-01-01 起的 Yahoo 日行情, 结束范围由 Yahoo 决定
5. 补齐每只证券首尾之间的内部缺行, 再写入 `prices`
6. 执行包内的 `factor_panel.sql`, 建立 `factor_panel` 视图

行情沿用现有入库脚本的 `auto_adjust=True`, OHLC 是复权价格
补齐日的价格沿用前值, 成交量为 0, 不补上市前或最后一条记录之后的数据
French 原始百分数除以 100 后入库, FF3 的 `rmw` 和 `cma` 为空
数据口径见 [数据源说明](../src/quanttoolkit/sources/USAGE.md) 和 [数据库说明](../src/quanttoolkit/db/README.md)

默认生成:

```text
当前工作目录/
├── logs/
│   ├── data_pipeline.log
│   └── trace.log
└── data/
    ├── quanttoolkit.db
    ├── ff3_daily.zip
    └── ff5_daily.zip
```

`data_pipeline.log` 追加记录文件夹位置, 步骤进度, 数据源日志, 入库条数与异常
`trace.log` 通过 [tracelog](tracelog.md) 记录函数耗时, 返回值和调用链
结束时打印调用链, 并卸下流水线的日志文件 sink
后续自己写的 `@trace` 函数继续使用这次设置的 `trace.log`
Yahoo 进度条仍由数据源显示在终端

## 再次运行与失败

每次重新获取名单, 两份 French 文件和全部范围的行情
`universe` 和 `ff_factors` 分别完整替换, `prices` 按 `(ticker, date)` 更新或追加
不会主动删除已经存下的其他股票历史, 再次运行也不会重复插入同一条行情

每张表各自事务写入, 失败的写入会回滚, 前面已经完成的步骤保留
某一步抛出异常时记录原因并停止, 修复后重新调用 `main()` 即可
部分证券没有抓到行情时记录缺失名单, 其余数据继续入库; 全部行情为空时停止
首次运行若中途失败, 数据库可能只有已经完成的表, 此时不能当作准备成功

`factor_panel` 从行情的第 64 个不同交易日起取数据, 同时要求当天有 FF5
French 发布进度可能落后于行情, 因此视图末日不一定等于行情末日
若两者还没有满足条件的共同日期, 视图可以为空
具体列与样本不足的行为见 [因子视图](../src/quanttoolkit/db/README.md#因子视图)

名单是抓取时的当前成员, 不是历史成分记录, 历史研究时需要考虑这个股票池口径
想更换日期范围, 股票池或复权设置时, 单独使用已有的数据源和数据库接口
