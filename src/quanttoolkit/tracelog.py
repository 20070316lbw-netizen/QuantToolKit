"""记录函数跑了多久, 返回什么, 以及先后调用了谁

在自己的步骤函数上加 @trace, 终端和 logs/trace.log 就会同时输出记录
默认日志路径相对当前工作目录, 可用 configure_log 改到其他位置
其他模块的日志不会写进这个文件
导入本模块时不会创建日志文件
"""

from __future__ import annotations

import atexit
import functools
import time
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_LOG_PATH = Path("logs") / "trace.log"
_DEFAULT_MAX_ROWS = 5
_REPR_LIMIT = 200
_TRACE_MARK = "tracelog"

__all__ = ["clear_log", "configure_log", "print_call_chain", "trace"]


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name
        self.elapsed = 0.0
        self.summary = ""
        self.children: list[_Node] = []


_roots: list[_Node] = []
_stack: ContextVar[tuple[_Node, ...]] = ContextVar(
    "quanttoolkit_tracelog_stack",
    default=(),
)
_sink_id: int | None = None
_chain_dirty = False
_log = logger.bind(**{_TRACE_MARK: True})


def _format_seconds(seconds: float, digits: int) -> str:
    return f"{seconds:.{digits}f}s"


def _truncate(text: str) -> str:
    if len(text) <= _REPR_LIMIT:
        return text
    return f"{text[:_REPR_LIMIT]}..."


def _contains_table(value: Any) -> bool:
    # 容器里的表格如果直接 repr, 多行内容会粘在键名后面, 再被截断
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return True
    if isinstance(value, dict):
        return any(_contains_table(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_table(item) for item in value)
    return False


def _indent(text: str) -> str:
    return "  " + text.replace("\n", "\n  ")


def _key_label(key: Any) -> str:
    if isinstance(key, str):
        return key
    return repr(key)


def _single_line(text: str) -> str:
    if "\n" not in text:
        return text
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _brief(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return f"DataFrame{value.shape}"
    if isinstance(value, pd.Series):
        return f"Series{value.shape}"
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}"
    if isinstance(value, (dict, list, tuple)) and _contains_table(value):
        return _truncate(_brief_container(value))
    return _truncate(_single_line(repr(value)))


def _brief_container(value: Any) -> str:
    if isinstance(value, dict):
        parts = [f"{key!r}: {_brief(item)}" for key, item in value.items()]
        return "{" + ", ".join(parts) + "}"
    parts = [_brief(item) for item in value]
    if isinstance(value, list):
        return "[" + ", ".join(parts) + "]"
    body = ", ".join(parts)
    if len(value) == 1:
        body += ","
    return "(" + body + ")"


def _brief_exception(exc: BaseException) -> str:
    return _truncate(f"{type(exc).__name__}: {exc}")


def _detail(value: Any, *, max_rows: int) -> str:
    if isinstance(value, pd.DataFrame):
        columns = [str(column) for column in value.columns]
        preview = value.head(max_rows).to_string()
        return f"DataFrame shape={value.shape} columns={columns}\n{preview}"
    if isinstance(value, pd.Series):
        preview = value.head(max_rows).to_string()
        return f"Series shape={value.shape} dtype={value.dtype}\n{preview}"
    if isinstance(value, np.ndarray):
        # 按行展开后取前几项, 和 max_rows 的预览长度对齐
        head = value.reshape(-1)[:max_rows].tolist()
        return f"ndarray shape={value.shape} dtype={value.dtype} head={head}"
    if isinstance(value, (dict, list, tuple)) and _contains_table(value):
        return _detail_container(value, max_rows=max_rows)
    return _truncate(repr(value))


def _detail_container(value: Any, *, max_rows: int) -> str:
    if isinstance(value, dict):
        items = [(_key_label(key), item) for key, item in value.items()]
    else:
        items = [(f"[{index}]", item) for index, item in enumerate(value)]
    blocks = [
        f"{label}:\n{_indent(_detail(item, max_rows=max_rows))}"
        for label, item in items
    ]
    return "\n\n".join(blocks)


def _append_text(text: str) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _file_sink(message: object) -> None:
    _append_text(str(message))


def _is_trace_record(record: dict[str, Any]) -> bool:
    # 只接收本工具打过标记的记录, 现有 logger 不会进 trace.log
    return bool(record["extra"].get(_TRACE_MARK))


def _ensure_sink() -> None:
    global _sink_id
    if _sink_id is not None:
        return
    _sink_id = logger.add(
        _file_sink,
        filter=_is_trace_record,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        diagnose=False,
        backtrace=True,
        colorize=False,
        catch=True,
    )


def _render_node(node: _Node, *, prefix: str, connector: str) -> list[str]:
    elapsed = _format_seconds(node.elapsed, 2)
    label = f"{node.name}  ({elapsed}) -> {node.summary}"
    lines = [f"{prefix}{connector}{label}"]
    if connector == "":
        child_prefix = ""
    elif connector.startswith("└"):
        child_prefix = f"{prefix}    "
    else:
        child_prefix = f"{prefix}│   "
    last = len(node.children) - 1
    for index, child in enumerate(node.children):
        branch = "└── " if index == last else "├── "
        lines.extend(_render_node(child, prefix=child_prefix, connector=branch))
    return lines


def _format_call_chain() -> str:
    lines = ["调用链:"]
    for root in _roots:
        lines.extend(_render_node(root, prefix="", connector=""))
    return "\n".join(lines)


def _call(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    max_rows: int,
) -> Any:
    global _chain_dirty
    _ensure_sink()
    node = _Node(fn.__name__)
    stack = _stack.get()
    # 栈顶是父调用, 根调用没有父节点
    if stack:
        stack[-1].children.append(node)
    else:
        _roots.append(node)
    _chain_dirty = True
    token = _stack.set((*stack, node))
    started = time.perf_counter()
    try:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            node.elapsed = elapsed
            node.summary = _brief_exception(exc)
            _log.opt(exception=exc).error(
                "{} 耗时 {} 抛出 {}",
                fn.__qualname__,
                _format_seconds(elapsed, 4),
                node.summary,
            )
            raise
        elapsed = time.perf_counter() - started
        node.elapsed = elapsed
        node.summary = _brief(result)
        detail = _detail(result, max_rows=max_rows)
        # 容器展开后从下一行开始, 避免表格首行粘在 "返回" 后面
        gap = "\n" if "\n" in detail and isinstance(result, (dict, list, tuple)) else " "
        _log.info(
            "{} 耗时 {} 返回{}{}",
            fn.__qualname__,
            _format_seconds(elapsed, 4),
            gap,
            detail,
        )
        return result
    finally:
        _stack.reset(token)


def trace(
    func: Callable[..., Any] | None = None,
    *,
    max_rows: int = _DEFAULT_MAX_ROWS,
) -> Callable[..., Any]:
    """装饰函数, 记录耗时, 返回值摘要和调用链

    不带参数时写 @trace
    需要限制预览长度时写 @trace(max_rows=10)

    函数正常返回后写一条日志, 内容是函数全名, 耗时和返回值摘要
    DataFrame 记录形状, 列名和前几行
    Series 与 ndarray 记录形状, dtype 和前几项
    dict, list, tuple 里含有 DataFrame 或 Series 时, 每个键或下标单独成段
    其他对象记录截断到 200 个字符的 repr
    调用链上的简述保持单行, 嵌套表格只写形状

    函数抛出异常时记录异常和耗时, 然后原样重新抛出
    调用关系放在 contextvars 里, 用来在退出时打印调用链

    Args:
        func: 被装饰的函数, 由 @trace 自动传入
        max_rows: 预览的行数或元素个数, 默认 5

    Returns:
        装饰后的函数
        带参数调用时返回装饰器
        函数的名字和文档通过 functools.wraps 保留

    Raises:
        原样重新抛出被装饰函数自己的异常
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _call(fn, args, kwargs, max_rows=max_rows)

        return wrapper

    if func is not None:
        return decorate(func)
    return decorate


def configure_log(path: str | Path) -> Path:
    """设置后续追踪记录和调用链的文件位置, 保留已有文件内容

    Args:
        path: 日志文件路径, 支持 str 或 Path, 展开用户目录符号
            相对路径以当前工作目录为准

    Returns:
        Path: 设置后的日志路径, 父目录不存在时自动创建
        后续 @trace, print_call_chain 和 clear_log() 都使用这个位置

    Raises:
        OSError: 日志目录无法创建

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> previous = _LOG_PATH
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     target = Path(directory) / "logs" / "trace.log"
        ...     configure_log(target) == target
        True
        >>> configure_log(previous) == previous
        True
    """
    global _LOG_PATH
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = target
    return target


def clear_log(path: str | Path | None = None) -> Path:
    """清空日志文件, 不存在时创建空文件

    适合放在脚本顶部, 避免和上一次运行的记录混在一起
    默认清空当前追踪文件, 初始位置为工作目录下的 logs/trace.log
    这个函数只动磁盘文件, 不挂上日志 sink

    Args:
        path: 要清空的文件, 默认使用 configure_log 设置的位置

    Returns:
        被清空的文件路径
    """
    target = _LOG_PATH if path is None else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    return target


def print_call_chain() -> str:
    """打印当前调用链, 并返回这段文本

    没有被追踪的调用时不打印
    进程结束时也会自动打印
    手动打印之后如果没有新的追踪调用, 退出时不再重复打印

    Returns:
        调用链文本
        没有记录时返回空字符串
    """
    global _chain_dirty
    _chain_dirty = False
    if not _roots:
        return ""
    text = _format_call_chain()
    print(text, flush=True)
    _append_text(f"{text}\n")
    return text


def _print_call_chain_at_exit() -> None:
    if not _chain_dirty:
        return
    print_call_chain()


def _reset() -> None:
    """清空内存中的调用链, 并卸下文件 sink

    测试用来隔离全局状态
    不删除已经写在磁盘上的日志
    """
    global _sink_id, _chain_dirty
    _roots.clear()
    _stack.set(())
    _chain_dirty = False
    if _sink_id is not None:
        logger.remove(_sink_id)
        _sink_id = None


# 进程退出时打印尚未输出的调用链, 导入阶段不创建日志文件
atexit.register(_print_call_chain_at_exit)
