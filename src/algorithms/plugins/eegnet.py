import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from braindecode.models import EEGNet
from ..base import BaseAlgorithm
from ..__init__ import register_algorithm


@register_algorithm
class EEGNetAlgorithm(BaseAlgorithm):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        # 初始化参数（兼容params格式，默认值适配BCICIV_2a）
        self.params = params or {}
        self.epochs = self.params.get('epochs', 50)
        self.batch_size = self.params.get('batch_size', 16)
        self.learning_rate = self.params.get('learning_rate', 1e-3)
        self.tolerance = self.params.get('tolerance', 1e-6)
        self.n_channels = self.params.get('n_channels', 22)  # BCICIV_2a默认22通道
        self.n_classes = self.params.get('n_classes', 4)  # BCICIV_2a是4分类（左/右/脚/舌）
        # 250Hz × 4s = 1000
        self.input_window_samples = self.params.get('input_window_samples', 1000)

        # 模型相关
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_names = None

    @property
    def name(self):
        """算法名称（兼容注册机制）"""
        return "eegnet"

    def get_params(self) -> Dict[str, Any]:
        """获取参数（兼容接口）"""
        return self.params

    def set_params(self, **params) -> 'EEGNetAlgorithm':
        """设置参数（兼容接口）"""
        self.params.update(params)
        self.epochs = self.params.get('epochs', 50)
        self.batch_size = self.params.get('batch_size', 16)
        self.learning_rate = self.params.get('learning_rate', 1e-3)
        self.tolerance = self.params.get('tolerance', 1e-6)
        self.n_channels = self.params.get('n_channels', 22)
        self.n_classes = self.params.get('n_classes', 4)
        self.input_window_samples = self.params.get('input_window_samples', 1000)
        return self

    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        # 保持 (N, C, T) 不动
        if len(X.shape) == 2:
            X = X.reshape(-1, self.n_channels, self.input_window_samples)

        X_tensor = torch.from_numpy(X).float().to(self.device)

        y_tensor = None
        if y is not None:
            y_tensor = torch.from_numpy(y).long().to(self.device)

        return X_tensor, y_tensor

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练EEGNet模型（兼容你的train接口，输入X/y为numpy数组）"""
        # 1. 数据预处理
        X_tensor, y_tensor = self._prepare_data(X, y)

        # 2. 初始化EEGNet模型
        self.model = EEGNet(
            n_chans=self.n_channels,  # ✅ 改名
            n_outputs=self.n_classes,  # ✅ 改名
            n_times=self.input_window_samples,  # ✅ 保留
        ).to(self.device)

        # 3. 优化器和损失函数（多分类）
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # 4. 构建数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # 5. 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # 早停判断
            avg_loss = total_loss / len(dataloader)
            if avg_loss < self.tolerance:
                print(f"EEGNet训练早停：epoch={epoch + 1}, 平均损失={avg_loss:.6f}")
                break

        print("EEGNet训练完成！")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测接口（兼容接口，返回numpy数组）"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        # 数据预处理
        X_tensor, _ = self._prepare_data(X)

        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        return y_pred.flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率（兼容多分类）"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train() 方法")

        # 数据预处理
        X_tensor, _ = self._prepare_data(X)

        # 预测概率
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_prob = torch.softmax(outputs, dim=1).cpu().numpy()
        return y_prob

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """评估接口（适配四分类）"""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        print("\n" + "=" * 50)
        print("           EEGNet算法评估结果")
        print("=" * 50)
        print(f"准确率      : {accuracy_score(y_test, y_pred):.4f}")
        print(f"精确率      : {precision_score(y_test, y_pred, zero_division=0, average='weighted'):.4f}")
        print(f"召回率      : {recall_score(y_test, y_pred, zero_division=0, average='weighted'):.4f}")
        print(f"F1分数      : {f1_score(y_test, y_pred, zero_division=0, average='weighted'):.4f}")
        print("\n混淆矩阵：")
        print(confusion_matrix(y_test, y_pred))
        print("=" * 50 + "\n")


def create_algorithm(params=None):
    """创建算法实例"""
    return EEGNetAlgorithm(params)