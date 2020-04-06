from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import torch
import numpy as np

random_state = 44

np.random.seed(random_state)


def load_data(N, n_features, n_informative, n_targets, coef, noise, random_state):
    X, y, true_coefs = make_regression(N, n_features,
                                       n_informative,
                                       n_targets,
                                       coef=coef,
                                       noise=noise,
                                       random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float().unsqueeze(1)
    y_test = torch.from_numpy(y_test).float().unsqueeze(1)

    return X_train, X_test, y_train, y_test


def load_toy_dataset(N=20, sigma=9):
    X = np.linspace(-4, 4, N)
    y = X**3 + np.random.randn(*X.shape) * sigma

    X_train = torch.from_numpy(X).float().unsqueeze(1)
    y_train = torch.from_numpy(y).float().unsqueeze(1)

    X_test = np.concatenate(
        (np.linspace(-6, -4, 10), np.linspace(4, 6, 10)), 0)
    y_test = X_test**3 + np.random.randn(*X_test.shape) * sigma

    X_test = torch.from_numpy(X_test).float().unsqueeze(1)
    y_test = torch.from_numpy(y_test).float().unsqueeze(1)

    return X_train, X_test, y_train, y_test


def classification_dataset(N, sigma, use_torch=False):
    mu1 = np.array([0, 0])*3
    mu2 = np.array([2, 0])*3
    mu3 = np.array([1, np.sqrt(3)])*3

    def cov(_alpha): return np.array([[1, 0], [0, 1]]) * _alpha

    alpha = sigma
    alpha1 = cov(alpha)
    alpha2 = cov(alpha)
    alpha3 = cov(alpha)

    def gen_normal(mu, alpha):
        return np.random.multivariate_normal(
            mu, alpha, size=N)

    x1 = gen_normal(mu1, alpha1)
    x2 = gen_normal(mu2, alpha2)
    x3 = gen_normal(mu3, alpha3)

    y1 = np.zeros((N, 1))
    y2 = np.ones((N, 1))
    y3 = np.ones((N, 1))*2

    x1 = np.concatenate((x1, y1), 1)
    x2 = np.concatenate((x2, y2), 1)
    x3 = np.concatenate((x3, y3), 1)

    X = np.vstack((x1, x2, x3))
    np.random.shuffle(X)

    X_ood = generate_ood(X.shape[0], X[:, 0:2], 5)

    if use_torch:
        return torch.from_numpy(X).float(), torch.from_numpy(X_ood).float()

    return X, X_ood


def generate_ood(N, x_train, dist):
    r = dist
    x_range = (x_train[:, 0].min()-r, x_train[:, 0].max()+r)
    v = np.random.uniform(x_range[0], x_range[1], size=(N, 2))
    return v


def split_dataset(X, X_ood, plot=False):

    train, test, X_ood_train, X_ood_test = train_test_split(X, X_ood)
    y_train = train[:, -1]
    y_test = test[:, -1]
    x_train = train[:, 0:2]
    x_test = test[:, 0:2]

    X_train = torch.from_numpy(train[:, 0:2]).float()
    y_train = torch.from_numpy(train[:, 2:]).long()
    X_test = torch.from_numpy(test[:, 0:2]).float()
    y_test = torch.from_numpy(test[:, 2:]).long()
    X_ood_train = torch.from_numpy(X_ood_train).float()
    X_ood_test = torch.from_numpy(X_ood_test).float()

    return X_train, y_train, X_test, y_test, X_ood_train, X_ood_test

# Generate training data


def generate_classification_data(N, sigma):
    X, X_ood = classification_dataset(N, sigma=sigma, use_torch=False)
    X_train, y_train, X_test, y_test, X_ood_train, X_ood_test = split_dataset(
        X, X_ood)
    return X_train, y_train, X_test, y_test, X_ood_train, X_ood_test

