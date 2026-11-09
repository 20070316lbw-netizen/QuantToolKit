"""MLP 的网络结构和排序训练参数"""

from dataclasses import dataclass, field

from quanttoolkit.model.params import ModelParams


@dataclass
class MLPParams(ModelParams):
    """MLP 截面排序配置, 可通过继承的方法保存为 JSON

    hidden_dims 是各隐藏层宽度, 空列表表示线性打分
    隐藏层使用 ReLU 和 dropout, 输出层为单个连续分数
    每个训练日期更新一次参数, max_epochs 表示完整遍历日期的轮数
    有验证集时始终恢复最佳权重, early_stopping_rounds=0 只关闭提前停止
    device 默认 cpu, 也可显式指定本机支持的 mps 或 cuda
    seed 控制网络初始化, dropout 和训练日期的打乱顺序
    相同 CPU 环境可复现, 不保证跨设备得到完全相同的结果

    Example:
        >>> params = MLPParams(hidden_dims=[16], seed=3)
        >>> params.to_dict()["hidden_dims"]
        [16]
        >>> params.max_epochs
        100
    """

    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 100
    early_stopping_rounds: int = 10
    device: str = "cpu"
