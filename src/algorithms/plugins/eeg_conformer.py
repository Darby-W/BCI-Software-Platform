import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from ..base import BaseAlgorithm
from ..__init__ import register_algorithm

# Conformer核心模型
class EEGConformerModel(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, seq_len=1000):
        super().__init__()

        # CNN提取局部特征
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU()
        )

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x)           # (B, 64, T)
        x = x.permute(0, 2, 1)    # (B, T, 64)
        x = self.transformer(x)   # (B, T, 64)
        x = x.mean(dim=1)         # 全局池化
        return self.classifier(x)

@register_algorithm
class EEGConformerAlgorithm(BaseAlgorithm):

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

        # 训练参数
        self.epochs = self.params.get('epochs', 20)
        self.batch_size = self.params.get('batch_size', 16)
        self.learning_rate = self.params.get('learning_rate', 1e-3)
        self.tolerance = self.params.get('tolerance', 1e-6)

        # 数据参数（BCICIV_2a默认）
        self.n_channels = self.params.get('n_channels', 22)
        self.n_classes = self.params.get('n_classes', 4)
        self.input_window_samples = self.params.get('input_window_samples', 1000)

        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 模型组件
        self.model = None
        self.optimizer = None
        self.criterion = None

    @property
    def name(self):
        return "eeg_conformer"

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def set_params(self, **params) -> 'EEGConformerAlgorithm':
        self.params.update(params)

        self.epochs = self.params.get('epochs', 20)
        self.batch_size = self.params.get('batch_size', 16)
        self.learning_rate = self.params.get('learning_rate', 1e-3)
        self.tolerance = self.params.get('tolerance', 1e-6)

        self.n_channels = self.params.get('n_channels', 22)
        self.n_classes = self.params.get('n_classes', 4)
        self.input_window_samples = self.params.get('input_window_samples', 1000)

        return self

    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        # 保持 (N, C, T)
        if len(X.shape) == 2:
            X = X.reshape(-1, self.n_channels, self.input_window_samples)

        X_tensor = torch.from_numpy(X).float().to(self.device)

        y_tensor = None
        if y is not None:
            y_tensor = torch.from_numpy(y).long().to(self.device)

        return X_tensor, y_tensor

    def train(self, X: np.ndarray, y: np.ndarray):

        # 数据
        X_tensor, y_tensor = self._prepare_data(X, y)

        # 初始化模型
        self.model = EEGConformerModel(
            n_channels=self.n_channels,
            n_classes=self.n_classes,
            seq_len=self.input_window_samples
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # 训练
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

            avg_loss = total_loss / len(dataloader)

            if avg_loss < self.tolerance:
                print(f"EEGConformer早停：epoch={epoch+1}, loss={avg_loss:.6f}")
                break

        print("EEGConformer训练完成！")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型未训练")

        X_tensor, _ = self._prepare_data(X)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        return y_pred.flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型未训练")

        X_tensor, _ = self._prepare_data(X)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_prob = torch.softmax(outputs, dim=1).cpu().numpy()

        return y_prob

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):

        y_pred = self.predict(X_test)

        print("\n" + "=" * 50)
        print("        EEGConformer评估结果")
        print("=" * 50)
        print(f"准确率  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"精确率  : {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"召回率  : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"F1分数  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print("\n混淆矩阵：")
        print(confusion_matrix(y_test, y_pred))
        print("=" * 50 + "\n")


def create_algorithm(params=None):
    return EEGConformerAlgorithm(params)