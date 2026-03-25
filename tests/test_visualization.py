"""
可视化模块测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pytest
from src.visualization.publication_plots import PublicationPlotter
from src.visualization.interactive_plots import InteractivePlotter


class TestPublicationPlotter:
    """论文级别图表测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.plotter = PublicationPlotter(output_dir="./test_figures")
        
        # 生成测试数据
        self.cm = np.array([[85, 15], [12, 88]])
        self.classes = ['Left', 'Right']
        
        # ROC数据
        self.fpr = np.linspace(0, 1, 50)
        self.tpr = self.fpr ** 0.5  # 模拟ROC曲线
        self.auc = 0.85
        
        # 学习曲线数据
        self.epochs = list(range(1, 51))
        self.train_scores = 0.5 + 0.4 * (1 - np.exp(-np.array(self.epochs) / 15))
        self.val_scores = 0.5 + 0.35 * (1 - np.exp(-np.array(self.epochs) / 20))
        
        # 特征重要性
        self.feature_names = [f"Feature_{i}" for i in range(15)]
        self.importance = np.random.rand(15)
    
    def test_plot_confusion_matrix(self):
        """测试混淆矩阵绘制"""
        fig = self.plotter.plot_confusion_matrix(
            self.cm,
            self.classes,
            title="Test Confusion Matrix",
            save=False
        )
        
        assert isinstance(fig, plt.Figure)
        assert fig.get_size_inches()[0] == 8  # 默认宽度
        plt.close(fig)
    
    def test_plot_confusion_matrix_normalized(self):
        """测试归一化混淆矩阵"""
        fig = self.plotter.plot_confusion_matrix(
            self.cm,
            self.classes,
            normalize=True,
            save=False
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_roc_curve_single(self):
        """测试单条ROC曲线"""
        fig = self.plotter.plot_roc_curve(
            self.fpr,
            self.tpr,
            self.auc,
            title="Test ROC Curve",
            save=False
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_roc_curve_multiple(self):
        """测试多条ROC曲线"""
        fpr_list = [self.fpr, self.fpr]
        tpr_list = [self.tpr, self.tpr ** 1.2]
        auc_list = [self.auc, 0.78]
        algorithms = ['Algorithm A', 'Algorithm B']
        
        fig = self.plotter.plot_roc_curve(
            fpr_list,
            tpr_list,
            auc_list,
            algorithms=algorithms,
            title="Test Multiple ROC Curves",
            save=False
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_accuracy_comparison(self):
        """测试算法对比图"""
        results = {
            'EEGNet': {'accuracy': 0.85, 'accuracy_std': 0.02},
            'DeepConvNet': {'accuracy': 0.88, 'accuracy_std': 0.025},
            'MI-EEGNet': {'accuracy': 0.895, 'accuracy_std': 0.018}
        }
        
        fig = self.plotter.plot_accuracy_comparison(
            results,
            metric='accuracy',
            title="Test Algorithm Comparison",
            save=False
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_learning_curves(self):
        """测试学习曲线绘制"""
        fig = self.plotter.plot_learning_curves(
            self.train_scores,
            self.val_scores,
            self.epochs,
            title="Test Learning Curves",
            save=False
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_feature_importance(self):
        """测试特征重要性图"""
        fig = self.plotter.plot_feature_importance(
            self.feature_names,
            self.importance,
            top_n=10,
            title="Test Feature Importance",
            save=False
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_save_figure(self):
        """测试图表保存"""
        fig = self.plotter.plot_confusion_matrix(
            self.cm,
            self.classes,
            save=True
        )
        
        # 验证文件是否创建
        import os
        assert os.path.exists(self.plotter.output_dir)
        plt.close(fig)


class TestInteractivePlotter:
    """交互式图表测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.plotter = InteractivePlotter(output_dir="./test_figures")
        
        # 生成测试数据
        self.evoked_data = {
            'Left': np.random.randn(32, 500),
            'Right': np.random.randn(32, 500)
        }
        self.times = np.linspace(-0.2, 0.8, 500)
        self.channels = [f'CH{i}' for i in range(32)]
        
        # 频谱数据
        self.spectra = {
            'Left': np.random.rand(50),
            'Right': np.random.rand(50)
        }
        self.frequencies = np.linspace(0, 40, 50)
        
        # 算法对比数据
        self.results_df = pd.DataFrame({
            'algorithm': ['EEGNet', 'DeepConvNet', 'MI-EEGNet'],
            'accuracy': [0.85, 0.88, 0.895],
            'accuracy_std': [0.02, 0.025, 0.018]
        })
        
        # 混淆矩阵
        self.cm = np.array([[85, 15], [12, 88]])
        self.classes = ['Left', 'Right']
    
    def test_plot_interactive_erp(self):
        """测试交互式ERP波形图"""
        fig = self.plotter.plot_interactive_erp(
            self.evoked_data,
            self.times,
            self.channels[:5],
            title="Test ERP Waveforms"
        )
        
        assert fig is not None
        # 检查图形是否有数据
        assert len(fig.data) > 0
    
    def test_plot_interactive_spectrum(self):
        """测试交互式频谱图"""
        fig = self.plotter.plot_interactive_spectrum(
            self.spectra,
            self.frequencies,
            title="Test Power Spectrum"
        )
        
        assert fig is not None
        assert len(fig.data) == 2  # 两个条件
    
    def test_plot_interactive_comparison(self):
        """测试交互式算法对比图"""
        fig = self.plotter.plot_interactive_comparison(
            self.results_df,
            metric='accuracy',
            title="Test Algorithm Comparison"
        )
        
        assert fig is not None
        assert len(fig.data) == 1  # 一个条形图
    
    def test_plot_interactive_confusion_matrix(self):
        """测试交互式混淆矩阵"""
        fig = self.plotter.plot_interactive_confusion_matrix(
            self.cm,
            self.classes,
            title="Test Confusion Matrix"
        )
        
        assert fig is not None
    
    def test_plot_interactive_learning_curve(self):
        """测试交互式学习曲线"""
        epochs = list(range(1, 51))
        train_scores = 0.5 + 0.4 * (1 - np.exp(-np.array(epochs) / 15))
        val_scores = 0.5 + 0.35 * (1 - np.exp(-np.array(epochs) / 20))
        
        fig = self.plotter.plot_interactive_learning_curve(
            train_scores,
            val_scores,
            epochs,
            title="Test Learning Curves"
        )
        
        assert fig is not None
        assert len(fig.data) == 2  # 训练和验证曲线
    
    def test_plot_interactive_3d_scatter(self):
        """测试交互式3D散点图"""
        # 生成特征数据
        n_samples = 100
        n_features = 10
        features = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, 2, n_samples)
        
        fig = self.plotter.plot_interactive_3d_scatter(
            features,
            labels,
            title="Test 3D Scatter"
        )
        
        assert fig is not None


if __name__ == "__main__":
    # 需要导入pandas
    import pandas as pd
    pytest.main([__file__])