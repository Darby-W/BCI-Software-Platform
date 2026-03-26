import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from ..base import BaseAlgorithm
from ..__init__ import register_algorithm

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("警告: XGBoost未安装")

@register_algorithm
class LogisticRegressionAlgorithm(BaseAlgorithm):
    """
    逻辑回归算法（针对小样本数据优化，使用随机森林+交叉验证）
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        
        # 使用随机森林代替XGBoost（更稳定）
        self.model_type = self.params.get('model_type', 'random_forest')  # 'random_forest' 或 'xgboost'
        
        # 随机森林参数（更适合小样本）
        self.n_estimators = self.params.get('n_estimators', 100)
        self.max_depth = self.params.get('max_depth', 5)
        self.min_samples_split = self.params.get('min_samples_split', 5)
        self.min_samples_leaf = self.params.get('min_samples_leaf', 2)
        
        # XGBoost参数（如果使用）
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.max_iter = self.params.get('max_iter', 100)
        self.subsample = self.params.get('subsample', 0.6)
        self.colsample_bytree = self.params.get('colsample_bytree', 0.6)
        
        # 正则化
        self.regularization = self.params.get('regularization', 'l2')
        self.reg_strength = self.params.get('reg_strength', 1.0)
        
        # PCA降维
        self.use_pca = self.params.get('use_pca', True)
        self.pca_components = self.params.get('pca_components', 15)  # 进一步降维
        
        # 特征处理
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = PCA(n_components=self.pca_components, random_state=42)
        
        # 模型参数
        self.weights = None
        self.bias = None
        self.num_classes = None
        self.is_multiclass = False
        self.is_fitted = False
        
        # 训练记录
        self.training_losses = []
        
        # 模型
        self._model = None

    @property
    def name(self):
        return "logistic_reg"

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def set_params(self, **params) -> 'LogisticRegressionAlgorithm':
        self.params.update(params)
        self.model_type = self.params.get('model_type', 'random_forest')
        self.n_estimators = self.params.get('n_estimators', 100)
        self.max_depth = self.params.get('max_depth', 5)
        self.min_samples_split = self.params.get('min_samples_split', 5)
        self.min_samples_leaf = self.params.get('min_samples_leaf', 2)
        self.use_pca = self.params.get('use_pca', True)
        self.pca_components = self.params.get('pca_components', 15)
        return self

    def _create_model(self):
        """创建模型"""
        if self.model_type == 'random_forest':
            # 随机森林更适合小样本
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                class_weight='balanced',  # 处理类别不平衡
                random_state=42,
                n_jobs=-1
            )
        else:
            # XGBoost
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost未安装")
            
            # 设置正则化
            if self.regularization == 'l1':
                reg_alpha = self.reg_strength
                reg_lambda = 0
            else:
                reg_alpha = 0
                reg_lambda = self.reg_strength
            
            model = xgb.XGBClassifier(
                n_estimators=self.max_iter,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        
        return model

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练模型"""
        print("\n" + "=" * 70)
        print("小样本优化算法（随机森林 + PCA降维）")
        print("=" * 70)
        
        # 数据预处理
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        self.is_multiclass = self.num_classes > 2
        
        n_samples, n_features = X.shape
        print(f"\n数据信息:")
        print(f"  样本数: {n_samples}")
        print(f"  原始特征数: {n_features}")
        print(f"  类别数: {self.num_classes}")
        
        # 类别分布
        class_counts = np.bincount(y_encoded)
        print(f"\n类别分布:")
        for i, count in enumerate(class_counts):
            print(f"  类别 {self.label_encoder.classes_[i]}: {count} 样本 ({count/n_samples*100:.1f}%)")
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA降维
        if self.use_pca and n_features > self.pca_components:
            print(f"\n执行PCA降维: {n_features} -> {self.pca_components}")
            X_scaled = self.pca.fit_transform(X_scaled)
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"  保留方差比例: {explained_var:.2%}")
        
        self.is_fitted = True
        
        # 使用交叉验证评估
        print(f"\n执行5折交叉验证...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model_temp = self._create_model()
        cv_scores = cross_val_score(model_temp, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # 训练最终模型
        print(f"\n训练最终模型...")
        self._model = self._create_model()
        self._model.fit(X_scaled, y_encoded)
        
        # 训练集准确率
        train_score = self._model.score(X_scaled, y_encoded)
        print(f"训练集准确率: {train_score:.4f}")
        
        # 检查过拟合
        if train_score - cv_scores.mean() > 0.2:
            print(f"⚠️  警告: 可能存在过拟合 (训练集 {train_score:.3f} vs 交叉验证 {cv_scores.mean():.3f})")
        
        # 特征重要性
        if hasattr(self._model, 'feature_importances_'):
            importances = self._model.feature_importances_
            top_indices = np.argsort(importances)[-5:][::-1]
            print(f"\n最重要的5个特征:")
            for idx in top_indices:
                print(f"  特征 {idx}: {importances[idx]:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        if self._model is None:
            raise ValueError("模型未训练")
        
        X = np.asarray(X, dtype=np.float64)
        if self.is_fitted:
            X = self.scaler.transform(X)
            if self.use_pca and hasattr(self, 'pca'):
                X = self.pca.transform(X)
        
        predictions = self._model.predict(X)
        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self._model is None:
            raise ValueError("模型未训练")
        
        X = np.asarray(X, dtype=np.float64)
        if self.is_fitted:
            X = self.scaler.transform(X)
            if self.use_pca and hasattr(self, 'pca'):
                X = self.pca.transform(X)
        
        proba = self._model.predict_proba(X)
        
        if self.num_classes == 2 and proba.shape[1] == 1:
            return np.hstack([1 - proba, proba])
        
        return proba

    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.predict(X_test)
        
        print("\n" + "=" * 70)
        print(f"模型评估结果 ({self.num_classes}分类)")
        print("=" * 70)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        if self.num_classes == 2:
            avg = 'binary'
            precision = precision_score(y_test, y_pred, zero_division=0, average=avg)
            recall = recall_score(y_test, y_pred, zero_division=0, average=avg)
            f1 = f1_score(y_test, y_pred, zero_division=0, average=avg)
        else:
            avg = 'weighted'
            precision = precision_score(y_test, y_pred, zero_division=0, average=avg)
            recall = recall_score(y_test, y_pred, zero_division=0, average=avg)
            f1 = f1_score(y_test, y_pred, zero_division=0, average=avg)
        
        print(f"\n性能指标:")
        print(f"  准确率 (Accuracy):  {accuracy:.4f}")
        print(f"  精确率 (Precision): {precision:.4f}")
        print(f"  召回率 (Recall):    {recall:.4f}")
        print(f"  F1分数 (F1-Score):  {f1:.4f}")
        
        print(f"\n混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # 各类别准确率
        print(f"\n各类别准确率:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            mask = (y_test == class_name)
            if np.sum(mask) > 0:
                class_acc = accuracy_score(y_test[mask], y_pred[mask])
                print(f"  类别 '{class_name}': {class_acc:.4f}")
        
        print("=" * 70 + "\n")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }


def create_algorithm(params=None):
    """创建算法实例"""
    return LogisticRegressionAlgorithm(params)