import numpy as np
from .base import BaseAlgorithm
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from . import register_algorithm

@register_algorithm
class DummyAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.model = DummyRegressor(strategy="mean")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def name(self) -> str:
        return "dummy"