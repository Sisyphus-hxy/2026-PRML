import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data import make_moons_3d


def build_models():
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)

    stump = DecisionTreeClassifier(max_depth=3, random_state=42)
    try:
        boost = AdaBoostClassifier(
            estimator=stump,
            n_estimators=200,
            learning_rate=1.0,
            random_state=42,
        )
    except TypeError:
        boost = AdaBoostClassifier(
            base_estimator=stump,
            n_estimators=200,
            learning_rate=1.0,
            random_state=42,
        )

    return {
        "Decision Tree": tree,
        "AdaBoost + DT": boost,
        "SVM (Linear)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0)),
        ]),
        "SVM (Poly)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="poly", degree=3, C=1.0, gamma="scale", coef0=1.0)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale")),
        ]),
    }


def plot_train_set(X, y):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis", s=18)
    ax.legend(*points.legend_elements(), title="Class")
    ax.set_title("Training set")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    plt.savefig("train_3d_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def evaluate(models, X_train, y_train, X_test, y_test):
    rows = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        rows.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, train_pred),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
        })

        print(f"\n{name}")
        print(classification_report(y_test, y_pred, digits=4))

    result = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    return result, predictions


def plot_confusion_matrices(y_test, predictions):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for idx, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=axes[idx], colorbar=False)
        axes[idx].set_title(name)

    for idx in range(len(predictions), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    X_train, y_train = make_moons_3d(n_samples=500, noise=0.2, random_state=42)
    X_test, y_test = make_moons_3d(n_samples=250, noise=0.2, random_state=2026)

    print("train:", X_train.shape, y_train.shape)
    print("test :", X_test.shape, y_test.shape)

    plot_train_set(X_train, y_train)

    result, predictions = evaluate(
        build_models(),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("\nsummary")
    print(result.to_string(index=False))
    result.to_csv("model_results.csv", index=False)

    plot_confusion_matrices(y_test, predictions)


if __name__ == "__main__":
    main()
