import numpy as np
from .base import BaseAlgorithm
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from . import register_algorithm

@register_algorithm
class BaselineAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.model = DummyClassifier(strategy="stratified")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def name(self) -> str:
        return "baseline"