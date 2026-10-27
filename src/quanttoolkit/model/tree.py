"""普通回归树与梯度提升树共用的建树逻辑"""

import numpy as np


class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.weight = None
        self.leaf_id = None


class RegressionTree:
    """基于平方误差的普通回归树

    无正则时, 每个叶子的预测值是该叶训练标签的均值
    apply 返回当前这棵树内部的叶子编号
    不同训练结果之间的叶子编号不表示同一类区域
    """

    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        lambda_l2=0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.lambda_l2 = lambda_l2
        self.root = None
        self.n_leaves = 0

    def _leaf_weight(self, g, h):
        return -g.sum() / (h.sum() + self.lambda_l2)

    def _gain(self, g_left, h_left, g_right, h_right):
        return 0.5 * (
            g_left**2 / (h_left + self.lambda_l2)
            + g_right**2 / (h_right + self.lambda_l2)
            - (g_left + g_right) ** 2
            / (h_left + h_right + self.lambda_l2)
        )

    def _best_split(self, X, g, h):
        best_feature = None
        best_threshold = None
        best_gain = 0.0

        for feature in range(X.shape[1]):
            # 最大值会产生空的右分支, 不作为候选
            thresholds = np.unique(X[:, feature])[:-1]

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue

                gain = self._gain(
                    g[left_mask].sum(),
                    h[left_mask].sum(),
                    g[right_mask].sum(),
                    h[right_mask].sum(),
                )

                if gain > best_gain:
                    best_feature = feature
                    best_threshold = threshold
                    best_gain = gain

        return best_feature, best_threshold

    def _make_leaf(self, g, h):
        node = TreeNode()
        node.weight = self._leaf_weight(g, h)
        node.leaf_id = self.n_leaves
        self.n_leaves += 1
        return node

    def _build_tree(self, X, g, h, depth):
        if (
            depth >= self.max_depth
            or len(g) < self.min_samples_split
            or len(g) < 2 * self.min_samples_leaf
        ):
            return self._make_leaf(g, h)

        feature, threshold = self._best_split(X, g, h)

        if feature is None:
            return self._make_leaf(g, h)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        node = TreeNode()
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(
            X[left_mask],
            g[left_mask],
            h[left_mask],
            depth + 1,
        )
        node.right = self._build_tree(
            X[right_mask],
            g[right_mask],
            h[right_mask],
            depth + 1,
        )
        return node

    def _fit_gradients(self, X, g, h):
        # GBDT 每轮传入当前损失的梯度和海森
        self.n_leaves = 0
        self.root = self._build_tree(X, g, h, depth=0)
        return self

    def fit(self, X, y):
        """拟合普通回归树

        Args:
            X: 二维数值数组, 形状为 [样本数, 特征数]
                所有值必须有限, 不支持缺失值
            y: 一维数值数组, 长度等于样本数
                所有值必须有限, 至少有一个样本

        Returns:
            当前模型, 每次调用都会重新建树

        Raises:
            ValueError: 输入形状不匹配, 样本为空或包含非有限值

        Example:
            >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
            >>> y = np.array([1.0, 1.0, 3.0, 3.0])
            >>> model = RegressionTree(max_depth=1).fit(X, y)
            >>> model.predict(X).tolist()
            [1.0, 1.0, 3.0, 3.0]
        """
        X, y = _prepare_xy(X, y)
        return self._fit_gradients(X, -y, np.ones_like(y))

    def _find_leaf(self, x):
        node = self.root

        while node.weight is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node

    def predict(self, X):
        """返回每条样本所在叶子的预测值

        Args:
            X: 二维有限数值数组, 特征顺序与训练时相同

        Returns:
            一维 ndarray, 单位与训练标签相同

        Raises:
            RuntimeError: 模型尚未训练

        Example:
            >>> model = RegressionTree().fit([[0], [1]], [2, 2])
            >>> model.predict([[0], [1]]).tolist()
            [2.0, 2.0]
        """
        if self.root is None:
            raise RuntimeError("请先调用 fit")

        X = np.asarray(X, dtype=float)
        return np.array([self._find_leaf(x).weight for x in X])

    def apply(self, X):
        """返回每条样本所在叶子的编号

        Args:
            X: 二维有限数值数组, 特征顺序与训练时相同

        Returns:
            一维整数 ndarray, 编号范围为 [0, n_leaves)
            编号只在当前已训练的树内有效

        Raises:
            RuntimeError: 模型尚未训练

        Example:
            >>> model = RegressionTree(max_depth=1)
            >>> model.fit([[0], [1], [2], [3]], [1, 1, 3, 3])
            ... # doctest: +ELLIPSIS
            <...RegressionTree object at ...>
            >>> model.apply([[0], [3]]).tolist()
            [0, 1]
        """
        if self.root is None:
            raise RuntimeError("请先调用 fit")

        X = np.asarray(X, dtype=float)
        return np.array(
            [self._find_leaf(x).leaf_id for x in X],
            dtype=int,
        )


def _prepare_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X 必须是二维数组, y 必须是一维数组")

    if len(y) == 0 or len(X) != len(y):
        raise ValueError("X 和 y 必须具有相同的非零样本数")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X 和 y 不能包含 NaN 或无穷值")

    return X, y