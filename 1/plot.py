import numpy as np
import matplotlib.pyplot as plt

def plot_polynomial_fits(train_x, train_y, test_x, test_y, theta_list, degree_list):
    """
    绘制不同degree的多项式拟合曲线

    Args:
        train_x: 训练数据 x
        train_y: 训练数据 y
        test_x: 测试数据 x
        test_y: 测试数据 y
        theta_list: theta参数列表，每个元素对应一个degree
        degree_list: degree列表
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 训练集图
    axes[0].scatter(train_x, train_y, alpha=0.5, s=30, label='Train data', color='black')

    # 测试集图
    axes[1].scatter(test_x, test_y, alpha=0.5, s=30, label='Test data', color='black')

    # 绘制每个degree的拟合曲线
    for idx, (degree, theta) in enumerate(zip(degree_list, theta_list)):
        color = colors[idx % len(colors)]

        # 训练集曲线
        x_smooth_train = np.linspace(min(train_x), max(train_x), 100)
        X_smooth_train = np.ones((len(x_smooth_train), 1))
        for i in range(1, degree + 1):
            X_smooth_train = np.column_stack((X_smooth_train, x_smooth_train**i))
        y_smooth_train = X_smooth_train @ theta
        train_mse = np.mean((np.column_stack([np.ones(len(train_x))] + [train_x**i for i in range(1, degree + 1)]) @ theta - train_y) ** 2)

        axes[0].plot(x_smooth_train, y_smooth_train, color=color, linewidth=2, label=f'degree={degree} (MSE={train_mse:.4f})')

        # 测试集曲线
        x_smooth_test = np.linspace(min(test_x), max(test_x), 100)
        X_smooth_test = np.ones((len(x_smooth_test), 1))
        for i in range(1, degree + 1):
            X_smooth_test = np.column_stack((X_smooth_test, x_smooth_test**i))
        y_smooth_test = X_smooth_test @ theta
        test_mse = np.mean((np.column_stack([np.ones(len(test_x))] + [test_x**i for i in range(1, degree + 1)]) @ theta - test_y) ** 2)

        axes[1].plot(x_smooth_test, y_smooth_test, color=color, linewidth=2, label=f'degree={degree} (MSE={test_mse:.4f})')

    # 设置训练集图
    train_x_min, train_x_max = min(train_x), max(train_x)
    train_x_center = (train_x_min + train_x_max) / 2
    train_x_range = (train_x_max - train_x_min) * 0.7
    axes[0].set_xlim([train_x_center - train_x_range, train_x_center + train_x_range])

    train_y_min, train_y_max = min(train_y), max(train_y)
    train_y_center = (train_y_min + train_y_max) / 2
    train_y_range = (train_y_max - train_y_min) * 0.7
    axes[0].set_ylim([train_y_center - train_y_range, train_y_center + train_y_range])

    axes[0].set_xlabel('x', fontsize=14)
    axes[0].set_ylabel('y', fontsize=14)
    axes[0].set_title('Training Data', fontsize=16)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=12)

    # 设置测试集图
    test_x_min, test_x_max = min(test_x), max(test_x)
    test_x_center = (test_x_min + test_x_max) / 2
    test_x_range = (test_x_max - test_x_min) * 0.7
    axes[1].set_xlim([test_x_center - test_x_range, test_x_center + test_x_range])

    test_y_min, test_y_max = min(test_y), max(test_y)
    test_y_center = (test_y_min + test_y_max) / 2
    test_y_range = (test_y_max - test_y_min) * 0.7
    axes[1].set_ylim([test_y_center - test_y_range, test_y_center + test_y_range])

    axes[1].set_xlabel('x', fontsize=14)
    axes[1].set_ylabel('y', fontsize=14)
    axes[1].set_title('Test Data', fontsize=16)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig('regression_fit.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_trig_basis_fits(train_x, train_y, test_x, test_y, theta_list, freq_list):
    """
    绘制不同频率的三角基函数拟合曲线
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 训练集图
    axes[0].scatter(train_x, train_y, alpha=0.5, s=30, label='Train data', color='black')

    # 测试集图
    axes[1].scatter(test_x, test_y, alpha=0.5, s=30, label='Test data', color='black')

    # 绘制每个频率的拟合曲线
    for idx, (num_freqs, theta) in enumerate(zip(freq_list, theta_list)):
        color = colors[idx % len(colors)]

        # 训练集曲线
        x_smooth_train = np.linspace(min(train_x), max(train_x), 100)
        n_smooth = len(x_smooth_train)
        X_smooth_train = np.ones((n_smooth, 1))
        X_smooth_train = np.column_stack((X_smooth_train, x_smooth_train))
        for k in range(1, num_freqs + 1):
            X_smooth_train = np.column_stack((X_smooth_train, np.sin(k * x_smooth_train)))
            X_smooth_train = np.column_stack((X_smooth_train, np.cos(k * x_smooth_train)))
        y_smooth_train = X_smooth_train @ theta

        # 计算训练集MSE
        X_train = np.ones((len(train_x), 1))
        X_train = np.column_stack((X_train, train_x))
        for k in range(1, num_freqs + 1):
            X_train = np.column_stack((X_train, np.sin(k * train_x)))
            X_train = np.column_stack((X_train, np.cos(k * train_x)))
        train_mse = np.mean((X_train @ theta - train_y) ** 2)

        axes[0].plot(x_smooth_train, y_smooth_train, color=color, linewidth=2, label=f'freq={num_freqs} (MSE={train_mse:.4f})')

        # 测试集曲线
        x_smooth_test = np.linspace(min(test_x), max(test_x), 100)
        n_smooth = len(x_smooth_test)
        X_smooth_test = np.ones((n_smooth, 1))
        X_smooth_test = np.column_stack((X_smooth_test, x_smooth_test))
        for k in range(1, num_freqs + 1):
            X_smooth_test = np.column_stack((X_smooth_test, np.sin(k * x_smooth_test)))
            X_smooth_test = np.column_stack((X_smooth_test, np.cos(k * x_smooth_test)))
        y_smooth_test = X_smooth_test @ theta

        # 计算测试集MSE
        X_test = np.ones((len(test_x), 1))
        X_test = np.column_stack((X_test, test_x))
        for k in range(1, num_freqs + 1):
            X_test = np.column_stack((X_test, np.sin(k * test_x)))
            X_test = np.column_stack((X_test, np.cos(k * test_x)))
        test_mse = np.mean((X_test @ theta - test_y) ** 2)

        axes[1].plot(x_smooth_test, y_smooth_test, color=color, linewidth=2, label=f'freq={num_freqs} (MSE={test_mse:.4f})')

    # 设置训练集图
    train_x_min, train_x_max = min(train_x), max(train_x)
    train_x_center = (train_x_min + train_x_max) / 2
    train_x_range = (train_x_max - train_x_min) * 0.7
    axes[0].set_xlim([train_x_center - train_x_range, train_x_center + train_x_range])

    train_y_min, train_y_max = min(train_y), max(train_y)
    train_y_center = (train_y_min + train_y_max) / 2
    train_y_range = (train_y_max - train_y_min) * 0.7
    axes[0].set_ylim([train_y_center - train_y_range, train_y_center + train_y_range])

    axes[0].set_xlabel('x', fontsize=14)
    axes[0].set_ylabel('y', fontsize=14)
    axes[0].set_title('Training Data (Trigonometric Basis)', fontsize=16)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=12)

    # 设置测试集图
    test_x_min, test_x_max = min(test_x), max(test_x)
    test_x_center = (test_x_min + test_x_max) / 2
    test_x_range = (test_x_max - test_x_min) * 0.7
    axes[1].set_xlim([test_x_center - test_x_range, test_x_center + test_x_range])

    test_y_min, test_y_max = min(test_y), max(test_y)
    test_y_center = (test_y_min + test_y_max) / 2
    test_y_range = (test_y_max - test_y_min) * 0.7
    axes[1].set_ylim([test_y_center - test_y_range, test_y_center + test_y_range])

    axes[1].set_xlabel('x', fontsize=14)
    axes[1].set_ylabel('y', fontsize=14)
    axes[1].set_title('Test Data (Trigonometric Basis)', fontsize=16)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig('trig_basis_fit.png', dpi=150, bbox_inches='tight')
    plt.show()

