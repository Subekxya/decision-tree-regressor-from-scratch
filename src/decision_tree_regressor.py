import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def compute_mse(y):
    """Calculate the mean squared error of a target subset."""

    if len(y) == 0:
        return 0

    mean = np.mean(y)

    return np.mean((y - mean) ** 2)


def compute_split_mse(y, left_mask, right_mask):
    """Calculate the weighted MSE after a proposed split."""

    y_left = y[left_mask]
    y_right = y[right_mask]

    n = len(y)

    weight_left = len(y_left) / n
    weight_right = len(y_right) / n

    return (
        weight_left * compute_mse(y_left)
        + weight_right * compute_mse(y_right)
    )


def leaf_value(y):
    """Calculate the prediction value for a leaf."""

    return np.mean(y)


def find_best_split(
    X,
    y,
    min_samples_leaf=1
):
    """Find the feature and threshold that minimize split MSE."""

    best_mse = float("inf")
    best_threshold = None
    best_feature = None

    _, n_features = X.shape

    for feature in range(n_features):

        feature_values = X[:, feature]

        unique_values = np.sort(
            np.unique(feature_values)
        )

        thresholds = (
            unique_values[:-1]
            + unique_values[1:]
        ) / 2

        for threshold in thresholds:

            left_mask = feature_values < threshold
            right_mask = feature_values >= threshold

            if np.sum(left_mask) < min_samples_leaf:
                continue

            if np.sum(right_mask) < min_samples_leaf:
                continue

            current_mse = compute_split_mse(
                y,
                left_mask,
                right_mask
            )

            if current_mse < best_mse:
                best_mse = current_mse
                best_threshold = threshold
                best_feature = feature

    return (
        best_feature,
        best_threshold,
        best_mse
    )


def build_tree(
    X,
    y,
    depth=0,
    max_depth=5,
    min_samples_split=3,
    min_samples_leaf=1
):
    """Recursively build the regression tree."""

    if depth >= max_depth:
        return Node(value=leaf_value(y))

    if len(y) < min_samples_split:
        return Node(value=leaf_value(y))

    (
        best_feature,
        best_threshold,
        _
    ) = find_best_split(
        X,
        y,
        min_samples_leaf
    )

    if best_threshold is None:
        return Node(value=leaf_value(y))

    left_mask = (
        X[:, best_feature] < best_threshold
    )

    right_mask = (
        X[:, best_feature] >= best_threshold
    )

    X_left = X[left_mask]
    y_left = y[left_mask]

    X_right = X[right_mask]
    y_right = y[right_mask]

    left_child = build_tree(
        X_left,
        y_left,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf
    )

    right_child = build_tree(
        X_right,
        y_right,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf
    )

    return Node(
        feature=best_feature,
        threshold=best_threshold,
        left=left_child,
        right=right_child
    )


def predict_one(node, x):
    """Predict the target value for one sample."""

    if node.value is not None:
        return node.value

    if x[node.feature] < node.threshold:
        return predict_one(node.left, x)

    return predict_one(node.right, x)


def predict(tree, X):
    """Predict target values for multiple samples."""

    predictions = []

    for x in X:
        predictions.append(
            predict_one(tree, x)
        )

    return np.array(predictions)


def evaluate_model(y_test, predictions):
    """Calculate regression evaluation metrics."""

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    print("Test MSE:", mse)
    print("Test RMSE:", rmse)
    print("Test R²:", r2)


def plot_predictions(y_test, predictions):
    """Plot actual values against predicted values."""

    plt.figure(figsize=(7, 5))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    min_value = min(
        y_test.min(),
        predictions.min()
    )

    max_value = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Diabetes Progression")

    plt.tight_layout()

    plt.savefig(
        "results/actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


def compare_with_sklearn(
    X_train,
    X_test,
    y_train,
    y_test
):
    """Compare the scratch implementation with scikit-learn."""

    sklearn_model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42
    )

    sklearn_model.fit(
        X_train,
        y_train
    )

    sklearn_predictions = sklearn_model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        sklearn_predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        sklearn_predictions
    )

    print("\nScikit-learn Decision Tree")
    print("Test MSE:", mse)
    print("Test RMSE:", rmse)
    print("Test R²:", r2)


def main():

    dataset = load_diabetes()

    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    tree = build_tree(
        X_train,
        y_train,
        max_depth=5,
        min_samples_split=3,
        min_samples_leaf=1
    )

    predictions = predict(
        tree,
        X_test
    )

    print("Scratch Decision Tree")
    evaluate_model(
        y_test,
        predictions
    )

    plot_predictions(
        y_test,
        predictions
    )

    compare_with_sklearn(
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()

