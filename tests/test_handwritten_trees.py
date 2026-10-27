import numpy as np
import pytest

from quanttoolkit.model import RegressionTree, SimpleGBDT


def test_tree_predict_and_apply():
    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([1.0, 1.0, 3.0, 3.0])

    model = RegressionTree(max_depth=1).fit(X, y)

    np.testing.assert_allclose(model.predict(X), y)
    np.testing.assert_array_equal(model.apply(X), [0, 0, 1, 1])


def test_min_samples_leaf_prevents_single_sample_leaf():
    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([0.0, 0.0, 0.0, 1.0])

    model = RegressionTree(
        max_depth=3,
        min_samples_leaf=2,
    ).fit(X, y)

    _, counts = np.unique(model.apply(X), return_counts=True)

    assert counts.min() >= 2


def test_constant_features_produce_one_leaf():
    X = np.ones((4, 2))
    y = np.array([1.0, 2.0, 3.0, 4.0])

    model = RegressionTree().fit(X, y)

    assert model.n_leaves == 1
    np.testing.assert_allclose(model.predict(X), y.mean())


def test_gbdt_refit_resets_trees():
    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = SimpleGBDT(n_trees=3, max_depth=1)

    model.fit(X, y)
    first_prediction = model.predict(X).copy()

    model.fit(X, y)

    assert len(model.trees) == 3
    np.testing.assert_allclose(model.predict(X), first_prediction)


def test_gbdt_learning_rate_scales_one_tree():
    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([1.0, 1.0, 3.0, 3.0])

    small = SimpleGBDT(n_trees=1, lr=0.1).fit(X, y)
    large = SimpleGBDT(n_trees=1, lr=0.5).fit(X, y)

    np.testing.assert_allclose(
        large.predict(X),
        5 * small.predict(X),
    )


def test_gbdt_early_stopping_and_refit_reset():
    X = np.arange(4.0).reshape(-1, 1)
    y = np.ones(4)
    y_val = np.zeros(4)

    model = SimpleGBDT(n_trees=20, lr=0.5)
    model.fit(
        X,
        y,
        X_val=X,
        y_val=y_val,
        early_stopping_rounds=2,
    )

    assert model.best_iteration == 1
    assert len(model.trees) == 1

    actual_loss = np.mean((model.predict(X) - y_val) ** 2)
    assert actual_loss == pytest.approx(model.best_val_loss)

    model.fit(X, y)

    assert len(model.trees) == 20
    assert model.best_iteration is None
    assert model.best_val_loss is None