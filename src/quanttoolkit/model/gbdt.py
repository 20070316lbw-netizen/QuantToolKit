"""基于平方误差的简化 GBDT 回归器"""

import numpy as np

from quanttoolkit.model.tree import RegressionTree, _prepare_xy


class SimpleGBDT:
    """通过逐轮拟合残差训练多棵回归树

    初始预测为 0, 每棵树的输出乘以学习率后累加
    提供验证集时, 训练结束后保留验证 MSE 最低的一轮
    """

    def __init__(
        self,
        n_trees=100,
        max_depth=3,
        lr=0.1,
        min_samples_split=2,
        min_samples_leaf=1,
        lambda_l2=0.0,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = lr
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.lambda_l2 = lambda_l2

        self.trees = []
        self.best_iteration = None
        self.best_val_loss = None
        self._fitted = False

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        early_stopping_rounds=None,
    ):
        """重新训练 GBDT

        Args:
            X: 二维有限数值数组, 形状为 [样本数, 特征数]
            y: 一维有限数值数组, 长度等于样本数
            X_val: 验证特征, 默认 None
                特征列数和顺序须与 X 一致
            y_val: 验证标签, 默认 None
                必须与 X_val 同时提供
            early_stopping_rounds: 连续多少轮未改善后停止
                默认 None, 表示跑完全部轮数
                设置时必须是正整数, 并提供验证集

        Returns:
            当前模型
            best_iteration 是保留的树数, 从 1 开始计数
            best_val_loss 是保留模型的验证 MSE
            没有验证集时, 这两个属性为 None

        Raises:
            ValueError: 输入不合法, 验证集不完整或早停设置不合法

        Example:
            >>> model = SimpleGBDT(n_trees=1, lr=1.0, max_depth=1)
            >>> _ = model.fit([[0], [1]], [1, 3])
            >>> model.predict([[0], [1]]).tolist()
            [1.0, 3.0]
        """
        X, y = _prepare_xy(X, y)

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val 和 y_val 必须同时提供")

        use_val = X_val is not None

        if early_stopping_rounds is not None and (
            not use_val or early_stopping_rounds < 1
        ):
            raise ValueError("早停需要验证集和正整数轮数")

        if use_val:
            X_val, y_val = _prepare_xy(X_val, y_val)
            if X_val.shape[1] != X.shape[1]:
                raise ValueError("训练集和验证集的特征列数必须相同")

        # 每次 fit 都重新训练
        self.trees = []
        self.best_iteration = None
        self.best_val_loss = None
        self._fitted = False

        y_pred = np.zeros(len(y), dtype=float)
        h = np.ones_like(y)
        rounds_no_improve = 0

        if use_val:
            y_val_pred = np.zeros(len(y_val), dtype=float)

        for _ in range(self.n_trees):
            g = y_pred - y

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                lambda_l2=self.lambda_l2,
            )
            tree._fit_gradients(X, g, h)

            self.trees.append(tree)
            y_pred += self.lr * tree.predict(X)

            if not use_val:
                continue

            y_val_pred += self.lr * tree.predict(X_val)
            val_loss = float(np.mean((y_val - y_val_pred) ** 2))

            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_iteration = len(self.trees)
                rounds_no_improve = 0
            else:
                rounds_no_improve += 1

                if (
                    early_stopping_rounds is not None
                    and rounds_no_improve >= early_stopping_rounds
                ):
                    break

        if use_val:
            self.trees = self.trees[:self.best_iteration]

        self._fitted = True
        return self

    def predict(self, X):
        """累加全部保留树的预测值

        Args:
            X: 二维有限数值数组, 特征顺序与训练时相同

        Returns:
            一维 ndarray, 单位与训练标签相同

        Raises:
            RuntimeError: 模型尚未训练

        Example:
            >>> model = SimpleGBDT(n_trees=1, lr=0.5)
            >>> _ = model.fit([[0], [1]], [2, 4])
            >>> model.predict([[0], [1]]).tolist()
            [1.0, 2.0]
        """
        if not self._fitted:
            raise RuntimeError("请先调用 fit")

        X = np.asarray(X, dtype=float)
        prediction = np.zeros(len(X), dtype=float)

        for tree in self.trees:
            prediction += self.lr * tree.predict(X)

        return prediction