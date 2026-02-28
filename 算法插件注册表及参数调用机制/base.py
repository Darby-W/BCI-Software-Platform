from abc import ABC, abstractmethod
import numpy as np

class BaseAlgorithm(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """算法名称，用于注册和查找"""
        pass