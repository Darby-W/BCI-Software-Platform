import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

class LogisticRegression:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """初始化逻辑回归算法
        
        Args:
            params: 算法参数
        """
        self.params = params or {}
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.max_iter = self.params.get('max_iter', 1000)
        self.tolerance = self.params.get('tolerance', 1e-6)
        self.weights = None
        self.bias = None
        self.feature_names = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数
        
        Args:
            z: 输入值
        
        Returns:
            激活值
        """
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LogisticRegression':
        """训练逻辑回归模型
        
        Args:
            X: 特征数据
            y: 目标变量
        
        Returns:
            self
        """
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 转换为numpy数组
        X_np = X.values
        y_np = y.values.reshape(-1, 1)
        
        # 初始化权重和偏置
        n_samples, n_features = X_np.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        # 梯度下降
        for i in range(self.max_iter):
            # 前向传播
            linear_model = np.dot(X_np, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X_np.T, (y_pred - y_np))
            db = (1 / n_samples) * np.sum(y_pred - y_np)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 检查收敛
            if np.linalg.norm(dw) < self.tolerance:
                break
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测
        
        Args:
            X: 特征数据
        
        Returns:
            预测结果
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # 前向传播
        X_np = X.values
        linear_model = np.dot(X_np, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        
        # 阈值处理
        return (y_pred >= 0.5).astype(int).flatten()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率
        
        Args:
            X: 特征数据
        
        Returns:
            预测概率
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # 前向传播
        X_np = X.values
        linear_model = np.dot(X_np, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        
        # 返回概率
        return np.hstack([1 - y_pred, y_pred])
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """获取算法参数
        
        Args:
            deep: 是否深度获取参数
        
        Returns:
            参数字典
        """
        return self.params
    
    def set_params(self, **params) -> 'LogisticRegression':
        """设置算法参数
        
        Args:
            params: 参数键值对
        
        Returns:
            self
        """
        self.params.update(params)
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.max_iter = self.params.get('max_iter', 1000)
        self.tolerance = self.params.get('tolerance', 1e-6)
        return self

# 插件接口
__all__ = ['LogisticRegression']

# 工厂函数
def create_algorithm(params: Optional[Dict[str, Any]] = None) -> LogisticRegression:
    """创建基线算法实例
    
    Args:
        params: 算法参数
    
    Returns:
        基线算法实例
    """
    return LogisticRegression(params)
