import numpy as np
from typing import Tuple, Optional
from .base import BaseSpatialFilter


class CSPFilter(BaseSpatialFilter): 
    def __init__(self, name: str = "csp", n_components: int = 4, reg: Optional[float] = None):

        super().__init__(name=name, n_components=n_components, reg=reg)
        self.reg = reg
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CSPFilter':

        # 验证输入
        if X.ndim != 3:
            raise ValueError(f"输入X必须是3维数组 (n_trials, n_channels, n_samples)，当前为{X.ndim}维")
        
        n_trials, n_channels, n_samples = X.shape
        
        # 检查标签
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(f"CSP需要二分类标签，当前有{len(unique_labels)}类: {unique_labels}")
        
        # 标准化数据（可选，但通常有助于稳定性）
        self.mean_ = np.mean(X, axis=(0, 2), keepdims=True)
        self.std_ = np.std(X, axis=(0, 2), keepdims=True) + 1e-8
        X_normalized = (X - self.mean_) / self.std_
        
        # 分离两类数据
        class_0_indices = np.where(y == unique_labels[0])[0]
        class_1_indices = np.where(y == unique_labels[1])[0]
        
        X0 = X_normalized[class_0_indices]  # 第一类: (n_trials_0, n_channels, n_samples)
        X1 = X_normalized[class_1_indices]  # 第二类: (n_trials_1, n_channels, n_samples)
        
        # 计算协方差矩阵
        cov_0 = self._compute_covariance(X0)
        cov_1 = self._compute_covariance(X1)
        
        # 求解广义特征值问题
        eigenvalues, eigenvectors = self._solve_generalized_eigenvalue(cov_0, cov_1)
        
        # 选择滤波器（从两端选择，最大化方差比）
        # 排序：大特征值对应第一类方差大、第二类方差小的成分
        #       小特征值对应第一类方差小、第二类方差大的成分
        sorted_indices = np.argsort(eigenvalues)[::-1]
        
        # 选择前n_components和后n_components
        selected_indices = np.concatenate([
            sorted_indices[:self.n_components],  # 第一类特征
            sorted_indices[-self.n_components:]   # 第二类特征
        ])
        
        self.filters_ = eigenvectors[:, selected_indices]  # (n_channels, 2*n_components)
        
        # 计算空间模式（用于可视化）
        self.patterns_ = np.linalg.pinv(self.filters_).T
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("CSP滤波器尚未训练，请先调用fit()方法")
        
        if X.ndim != 3:
            raise ValueError(f"输入X必须是3维数组 (n_trials, n_channels, n_samples)，当前为{X.ndim}维")
        
        # 标准化
        X_normalized = (X - self.mean_) / self.std_
        
        n_trials = X.shape[0]
        n_filters = self.filters_.shape[1]
        
        # 应用空间滤波
        features = np.zeros((n_trials, n_filters))
        
        for i in range(n_trials):
            # 应用滤波器: W^T * X
            filtered = self.filters_.T @ X_normalized[i]  # (2*n_components, n_samples)
            
            # 计算对数方差作为特征
            var = np.var(filtered, axis=1)
            features[i] = np.log(var + 1e-8)  # 加小值避免log(0)
        
        return features
    
    def _apply_filter(self, X: np.ndarray) -> np.ndarray:
        # 标准化
        X_normalized = (X - self.mean_) / self.std_
        
        n_trials = X.shape[0]
        n_samples = X.shape[2]
        n_filters = self.filters_.shape[1]
        
        filtered_signal = np.zeros((n_trials, n_filters, n_samples))
        
        for i in range(n_trials):
            filtered_signal[i] = self.filters_.T @ X_normalized[i]
        
        return filtered_signal
    
    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        n_trials, n_channels, n_samples = X.shape
        cov_sum = np.zeros((n_channels, n_channels))
        
        for i in range(n_trials):
            trial = X[i]
            # 计算单个试次的协方差矩阵
            cov_trial = trial @ trial.T / np.trace(trial @ trial.T)
            cov_sum += cov_trial
        
        return cov_sum / n_trials
    
    def _solve_generalized_eigenvalue(self, cov_0: np.ndarray, cov_1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 添加正则化以提高数值稳定性
        reg = self.reg if self.reg is not None else 0.01
        
        n_channels = cov_0.shape[0]
        cov_1_reg = cov_1 + reg * np.eye(n_channels) * np.trace(cov_1) / n_channels
        
        # 求解广义特征值问题
        try:
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(cov_1_reg) @ cov_0)
            
            # 转换为实数（去除数值误差导致的虚部）
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
            
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用更稳定的伪逆
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(cov_1_reg) @ cov_0)
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
        
        return eigenvalues, eigenvectors


# 便捷函数
def compute_csp_features(X: np.ndarray, y: np.ndarray, n_components: int = 4, reg: Optional[float] = None) -> Tuple[np.ndarray, CSPFilter]:
    csp = CSPFilter(n_components=n_components, reg=reg)
    features = csp.fit_transform(X, y)
    return features, csp
