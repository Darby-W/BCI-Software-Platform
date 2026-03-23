import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from ..base import BaseAlgorithm
from ..__init__ import register_algorithm

@register_algorithm
class LogisticRegressionAlgorithm(BaseAlgorithm):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.max_iter = self.params.get('max_iter', 1000)
        self.tolerance = self.params.get('tolerance', 1e-6)
        self.weights = None
        self.bias = None
        self.feature_names = None
        self.num_classes = None
        self.is_multiclass = False
        # 添加正则化参数
        self.regularization = self.params.get('regularization', 'l2')  # 'l1', 'l2', or None
        self.reg_strength = self.params.get('reg_strength', 0.01)

    @property
    def name(self):
        return "logistic_reg"

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def set_params(self, **params) -> 'LogisticRegressionAlgorithm':
        self.params.update(params)
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.max_iter = self.params.get('max_iter', 1000)
        self.tolerance = self.params.get('tolerance', 1e-6)
        self.regularization = self.params.get('regularization', 'l2')
        self.reg_strength = self.params.get('reg_strength', 0.01)
        return self

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid函数，用于二分类"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax函数，用于多分类
        输入: z shape (n_samples, n_classes)
        输出: 概率分布 shape (n_samples, n_classes)
        """
        # 数值稳定性处理：减去每行的最大值
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_loss(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算损失函数（交叉熵损失 + 正则化）
        """
        n_samples = X.shape[0]
        
        # 交叉熵损失
        if self.is_multiclass:
            # 多分类：使用softmax的交叉熵
            y_one_hot = np.eye(self.num_classes)[y.flatten()]
            loss = -np.mean(np.sum(y_one_hot * np.log(y_pred + 1e-10), axis=1))
        else:
            # 二分类：使用sigmoid的交叉熵
            y = y.flatten()
            loss = -np.mean(y * np.log(y_pred.flatten() + 1e-10) + 
                          (1 - y) * np.log(1 - y_pred.flatten() + 1e-10))
        
        # 添加正则化项
        if self.regularization == 'l2':
            loss += (self.reg_strength / (2 * n_samples)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            loss += (self.reg_strength / n_samples) * np.sum(np.abs(self.weights))
        
        return loss

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple:
        """
        计算梯度和偏置梯度
        """
        n_samples = X.shape[0]
        
        if self.is_multiclass:
            # 多分类：使用softmax的梯度
            y_one_hot = np.eye(self.num_classes)[y.flatten()]
            error = y_pred - y_one_hot
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error, axis=0)
            
            # 添加正则化梯度
            if self.regularization == 'l2':
                dw += (self.reg_strength / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.reg_strength / n_samples) * np.sign(self.weights)
        else:
            # 二分类：使用sigmoid的梯度
            error = y_pred - y.reshape(-1, 1)
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)
            
            # 添加正则化梯度
            if self.regularization == 'l2':
                dw += (self.reg_strength / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.reg_strength / n_samples) * np.sign(self.weights)
        
        return dw, db

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练模型，自动检测类别数量，支持二分类和多分类
        """
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)
        
        # 检测类别数量
        self.num_classes = len(np.unique(y))
        self.is_multiclass = self.num_classes > 2
        
        # 初始化参数
        if self.is_multiclass:
            # 多分类：weights shape (n_features, n_classes)
            self.weights = np.zeros((n_features, self.num_classes))
            self.bias = np.zeros(self.num_classes)
        else:
            # 二分类：weights shape (n_features, 1)
            self.weights = np.zeros((n_features, 1))
            self.bias = 0
        
        # 添加Adam优化器参数
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        m_w = np.zeros_like(self.weights)
        v_w = np.zeros_like(self.weights)
        m_b = np.zeros_like(self.bias)
        v_b = np.zeros_like(self.bias)
        t = 0
        
        # 训练循环
        prev_loss = float('inf')
        for i in range(self.max_iter):
            t += 1
            
            # 前向传播
            linear = X @ self.weights + self.bias
            
            if self.is_multiclass:
                y_pred = self._softmax(linear)
            else:
                y_pred = self._sigmoid(linear)
            
            # 计算损失
            current_loss = self._compute_loss(X, y, y_pred)
            
            # 早停条件
            if abs(prev_loss - current_loss) < self.tolerance:
                break
            prev_loss = current_loss
            
            # 计算梯度
            dw, db = self._compute_gradients(X, y, y_pred)
            
            # Adam优化器更新参数
            m_w = beta1 * m_w + (1 - beta1) * dw
            v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
            m_b = beta1 * m_b + (1 - beta1) * db
            v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
            
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            
            self.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
            
            # 可选：打印训练进度（每100次迭代）
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.max_iter}, Loss: {current_loss:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        """
        if self.weights is None:
            raise ValueError("模型未训练，请先调用 train()")
        
        linear_model = np.dot(X, self.weights) + self.bias
        
        if self.is_multiclass:
            # 多分类：返回概率最大的类别
            y_pred = self._softmax(linear_model)
            return np.argmax(y_pred, axis=1)
        else:
            # 二分类：使用sigmoid阈值
            y_pred = self._sigmoid(linear_model)
            return (y_pred >= 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率分布
        """
        if self.weights is None:
            raise ValueError("模型未训练，请先调用 train()")
        
        linear_model = np.dot(X, self.weights) + self.bias
        
        if self.is_multiclass:
            # 多分类：返回所有类别的概率
            return self._softmax(linear_model)
        else:
            # 二分类：返回 [P(class=0), P(class=1)]
            prob_class_1 = self._sigmoid(linear_model).reshape(-1, 1)
            return np.hstack([1 - prob_class_1, prob_class_1])

    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        """
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        print("\n" + "=" * 50)
        print(f"           逻辑回归算法评估结果 ({self.num_classes}分类)")
        print("=" * 50)
        
        # 选择适当的average参数
        if self.num_classes == 2:
            avg = 'binary'
        else:
            avg = 'weighted'
        
        print(f"准确率      : {accuracy_score(y_test, y_pred):.4f}")
        print(f"精确率      : {precision_score(y_test, y_pred, zero_division=0, average=avg):.4f}")
        print(f"召回率      : {recall_score(y_test, y_pred, zero_division=0, average=avg):.4f}")
        print(f"F1分数      : {f1_score(y_test, y_pred, zero_division=0, average=avg):.4f}")
        
        print("\n混淆矩阵：")
        print(confusion_matrix(y_test, y_pred))
        print("=" * 50 + "\n")
        
        # 返回评估指标字典
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0, average=avg),
            'recall': recall_score(y_test, y_pred, zero_division=0, average=avg),
            'f1': f1_score(y_test, y_pred, zero_division=0, average=avg),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

def create_algorithm(params=None):
    """
    创建算法实例的工厂函数
    """
    return LogisticRegressionAlgorithm(params)