"""把模型参数和训练记录存成 JSON, 需要时再读回来"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelParams:
    """各模型共用的参数基类

    子类只增加自己的字段
    字段使用 str, int, float, bool, list, dict, 以便写成 JSON
    seed 是各模型共用的随机种子

    Example:
        >>> ModelParams(seed=1).to_dict()
        {'seed': 1}
    """

    seed: int = 7

    def to_dict(self) -> dict:
        """返回一份新的字段字典

        Returns:
            包含当前全部字段的 dict
            修改返回值不影响原对象
        """
        return asdict(self)

    def save(self, path: Path) -> None:
        """把参数写成 JSON 文件

        Args:
            path: 输出路径, 父目录需要已经存在
        """
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> ModelParams:
        """从 JSON 文件读回当前类的参数

        Args:
            path: 由 save 写出的 JSON 路径

        Returns:
            当前类的实例

        Raises:
            TypeError: JSON 的字段与当前类对不上
        """
        return cls(**json.loads(path.read_text()))


@dataclass
class TrainLog:
    """一次训练的模型名, 参数和指标

    params 使用 ModelParams.to_dict() 的结果
    metrics 在训练完成前可以是空字典

    Example:
        >>> TrainLog(model="lightgbm", params={"seed": 7}).metrics
        {}
    """

    model: str
    params: dict
    metrics: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """把训练日志写成 JSON 文件

        Args:
            path: 输出路径, 父目录需要已经存在
        """
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> TrainLog:
        """从 JSON 文件读回训练日志

        Args:
            path: 由 save 写出的 JSON 路径

        Returns:
            TrainLog

        Raises:
            TypeError: JSON 的字段与 TrainLog 对不上
        """
        return cls(**json.loads(path.read_text()))
