# 函数追踪

实现见 [tracelog.py](../src/quanttoolkit/tracelog.py)。
`@trace` 在函数返回后写一条记录, 内容是函数全名, 耗时和返回值摘要。
进程结束时自动打印一棵调用链, 里面只有挂了装饰器的函数。
终端和当前工作目录下的 `logs/trace.log` 同时输出, 文件按追加写入。
这条文件日志使用独立的 loguru sink, 并用 extra 标记过滤, 其他模块的 logger 不会写进来。
导入模块不会创建日志文件, 第一次追踪时才挂上文件输出。

## 用法

在脚本顶部调用 `clear_log()`, 把上一次的 `logs/trace.log` 清成空文件。
文件不存在时会连目录一起创建。
传入 `path` 时清空别的文件, 返回实际清空的路径。

`@trace` 和 `@trace(max_rows=10)` 都可以。
`max_rows` 默认 5, 控制 DataFrame 预览行数, 以及 Series 和 ndarray 预览的元素个数。
装饰器通过 `functools.wraps` 保留函数名字和文档。

调用关系放在 `contextvars` 里。
父调用是栈顶那个同样挂了 `@trace` 的函数, 中间没装饰的函数不会出现在树上。
没有被追踪的调用时, 退出不打印调用链。
脚本中途可以调用 `print_call_chain()` 立刻打印, 并拿到同一段文本。
手动打印之后如果没有新的追踪调用, 退出时不会再打一遍。

函数抛出异常时, 日志记下异常类型, 信息和耗时, 调用链节点用异常简述代替返回值, 然后原样重新抛出。

## 返回值摘要

| 类型 | 日志里的内容 | 调用链上的简述 |
| --- | --- | --- |
| DataFrame | 形状, 列名, 前 `max_rows` 行 | `DataFrame(行, 列)` |
| Series | 形状, dtype, 前 `max_rows` 项 | `Series(长度,)` |
| ndarray | 形状, dtype, 按行展开后的前 `max_rows` 项 | `ndarray` 加形状 |
| 其他对象 | 截断到 200 个字符的 repr | 同样的截断 repr |

调用链每个节点显示函数名, 耗时和上面的简述。
耗时包含该函数内部的子调用。
下面是一棵树的样子, 耗时会随机器变化:

```text
调用链:
main  (0.01s) -> ndarray(3,)
├── load_prices  (0.00s) -> DataFrame(3, 2)
├── build_factor  (0.00s) -> Series(3,)
│   └── zscore  (0.00s) -> Series(3,)
└── to_array  (0.00s) -> ndarray(3,)
```

追踪记录由 loguru 打到终端, 并追加写入日志文件。
调用链打印到标准输出, 同时追加到同一个文件。

## 可独立运行的示例

```python
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from quanttoolkit.tracelog import clear_log, print_call_chain, trace

origin = os.getcwd()
with tempfile.TemporaryDirectory() as directory:
    os.chdir(directory)
    try:
        log_path = clear_log()
        assert log_path == Path("logs") / "trace.log"
        assert log_path.read_text(encoding="utf-8") == ""

        @trace
        def load_prices():
            return pd.DataFrame(
                {
                    "close": [10.0, 11.0, 12.0],
                    "volume": [100, 110, 120],
                }
            )

        @trace
        def zscore(values):
            centered = values - values.mean()
            return centered / values.std(ddof=1)

        @trace(max_rows=2)
        def build_factor(prices):
            return zscore(prices["close"])

        @trace
        def to_array(values):
            return np.asarray(values)

        @trace
        def main():
            prices = load_prices()
            factor = build_factor(prices)
            return to_array(factor)

        result = main()
        assert result.shape == (3,)

        logger.info("其他模块的日志")
        text = print_call_chain()
        assert text.splitlines()[0] == "调用链:"
        assert "main  (" in text
        assert "├── load_prices" in text
        assert "├── build_factor" in text
        assert "│   └── zscore" in text
        assert "└── to_array" in text
        assert "DataFrame(3, 2)" in text
        assert "Series(3,)" in text
        assert "ndarray(3,)" in text

        written = log_path.read_text(encoding="utf-8")
        assert "load_prices" in written
        assert "close" in written
        assert "volume" in written
        assert "float64" in written
        assert "其他模块的日志" not in written
    finally:
        os.chdir(origin)
```
