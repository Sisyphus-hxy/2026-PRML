import matplotlib.pyplot as plt
import numpy as np


def make_moons_3d(n_samples=500, noise=0.1, random_state=None):
    rng = np.random.default_rng(random_state)

    theta = np.linspace(0, np.pi, n_samples)
    moon_a = np.column_stack([
        np.cos(theta),
        np.sin(theta),
        0.5 * np.sin(2 * theta),
    ])
    moon_b = np.column_stack([
        1 - np.cos(theta),
        1 - np.sin(theta) - 0.5,
        -0.5 * np.sin(2 * theta),
    ])

    X = np.vstack([moon_a, moon_b])
    y = np.hstack([
        np.zeros(n_samples, dtype=int),
        np.ones(n_samples, dtype=int),
    ])

    X += rng.normal(0, noise, size=X.shape)
    return X, y


if __name__ == "__main__":
    X, y = make_moons_3d(n_samples=500, noise=0.2, random_state=42)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis", s=18)
    ax.legend(*points.legend_elements(), title="Class")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D moon data")

    plt.tight_layout()
    plt.savefig("data_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
