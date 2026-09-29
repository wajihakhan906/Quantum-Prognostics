"""Classical baselines, including an Extreme Learning Machine (ELM)."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR


class ELMClassifier(BaseEstimator, ClassifierMixin):
    """Single hidden layer with random weights; output weights by regularised least squares."""

    def __init__(self, hidden=200, reg=1e-3, random_state=0):
        self.hidden, self.reg, self.random_state = hidden, reg, random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        self.W_ = rng.standard_normal((X.shape[1], self.hidden))
        self.b_ = rng.standard_normal(self.hidden)
        H = np.tanh(X @ self.W_ + self.b_)
        T = (y[:, None] == self.classes_[None]).astype(float)
        self.beta_ = np.linalg.solve(H.T @ H + self.reg * np.eye(self.hidden), H.T @ T)
        return self

    def predict(self, X):
        return self.classes_[np.argmax(np.tanh(X @ self.W_ + self.b_) @ self.beta_, 1)]


def classifiers():
    return {
        "SVM (RBF)": SVC(C=10),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=0),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0),
        "ELM": ELMClassifier(),
        "Softmax": LogisticRegression(max_iter=2000),
    }


def regressors():
    return {
        "SVR (RBF)": SVR(C=10, epsilon=0.02),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=0),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=0),
    }
