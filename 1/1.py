import numpy as np
import pandas as pd
from plot import plot_polynomial_fits, plot_trig_basis_fits

def fit_trig_basis(X_origin, y, num_freqs=3):
    n = len(X_origin)
    X = np.ones((n, 1))
    X = np.column_stack((X, X_origin))

    for k in range(1, num_freqs + 1):
        X = np.column_stack((X, np.sin(k * X_origin)))
        X = np.column_stack((X, np.cos(k * X_origin)))

    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    y_pred = X @ theta
    mse = np.mean((y_pred - y) ** 2)

    print(f"trig_basis(freq={num_freqs}) MSE: {mse}")
    return theta, mse

def test_trig_basis(X_test, y, theta, num_freqs=3):
    
    n = len(X_test)
    X = np.ones((n, 1))
    X = np.column_stack((X, X_test))

    for k in range(1, num_freqs + 1):
        X = np.column_stack((X, np.sin(k * X_test)))
        X = np.column_stack((X, np.cos(k * X_test)))

    y_pred = X @ theta
    mse = np.mean((y_pred - y) ** 2)
    print(f"Test trig_basis(freq={num_freqs}) MSE: {mse}")
    return mse

def fit_least_squares(X, y):
    ones = np.ones(len(X))
    X = np.column_stack((ones, X))
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    b = theta[0]
    w = theta[1]
    y_pred = b + w * X[:, 1]
    mse = np.mean((y_pred - y) ** 2)
    print("square_b:", b)
    print("square_w:", w)
    print("square_MSE:", mse)
    return b,w,mse

def fit_gradient_descent(X, y, lr, max_iter):
    b, w = 0, 0
    for i in range(max_iter):
        y_pred = b + w * X
        error = y_pred - y
        db = 2 * np.mean(error)
        dw = 2 * np.mean(error * X)
        b -= lr * db
        w -= lr * dw
    mse = np.mean((b + w * X - y) ** 2)

    print("grad_b:", b)
    print("grad_w:", w)
    print("grad_MSE:", mse)
    return b, w ,mse

def fit_newton(X_train, y_train):
    n = len(X_train)
    X = np.column_stack((np.ones(n), X_train))
    theta = np.zeros(2)
    H = (2/n) * X.T @ X
    g = (2/n) * X.T @ (X @ theta - y_train)

    theta -= np.linalg.inv(H) @ g

    b, w = theta
    mse = np.mean((b + w * X_train - y_train) ** 2)
    print("newton_b:", b)
    print("newton_w:", w)
    print("newton_mse:", mse)
    return b , w , mse

def fit_degree(X_origin, y, degree):
   
    X = np.ones(len(X_origin))
    for i in range(1, degree + 1):
        X = np.column_stack((X, X_origin**i))

    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    
    for i in range(degree + 1):
        print(f"degree={degree}_theta[{i}]:", theta[i])

    y_pred = X @ theta
    mse = np.mean((y_pred - y) ** 2)
    print(f"degree={degree}_MSE:", mse)
    return theta ,mse

def test_degree(test_x, y, theta, degree):
    X_test = np.ones((len(test_x), 1))
    for i in range(1, degree + 1):
        X_test = np.column_stack((X_test, test_x**i))

    y_pred = X_test @ theta
    mse = np.mean((y_pred - y) ** 2)
    print("Test MSE:", mse)
    print("\n")
    return mse


def test(X, y, b, w):
    y_pred = b + w * X
    mse = np.mean((y_pred - y) ** 2)
    print("Test MSE:", mse)
    print("\n")
    return mse




if __name__ == "__main__":
    # dataloader
    train_data = pd.read_csv("Data4Regression - Training Data.csv")
    train_x = train_data["x"].values
    train_y = train_data["y_complex"].values

    test_data = pd.read_csv("Data4Regression - Test Data.csv")
    test_x = test_data["x_new"].values
    test_y = test_data["y_new_complex"].values

    b, w, mse = fit_least_squares(train_x, train_y)
    test(test_x, test_y, b, w)

    b, w, mse = fit_gradient_descent(train_x, train_y, lr=0.01, max_iter=1000)
    test(test_x, test_y, b, w)

    b, w, mse = fit_newton(train_x, train_y)
    test(test_x, test_y, b, w)

    degree = 12
    theta, mse = fit_degree(train_x, train_y, degree)
    test_degree(test_x, test_y, theta, degree)

    degree_list = [1, 2, 3, 4, 5, 12]
    theta_list = []

    for degree in degree_list:
        theta, mse = fit_degree(train_x, train_y, degree)
        theta_list.append(theta)

    plot_polynomial_fits(train_x, train_y, test_x, test_y, theta_list, degree_list)

    print("\n")
    freq_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    theta_trig_list = []

    for num_freqs in freq_list:
        theta, mse = fit_trig_basis(train_x, train_y, num_freqs)
        theta_trig_list.append(theta)

    plot_trig_basis_fits(train_x, train_y, test_x, test_y, theta_trig_list, freq_list)
