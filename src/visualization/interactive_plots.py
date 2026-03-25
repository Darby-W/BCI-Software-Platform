"""
交互式图表模块
使用Plotly实现可交互的可视化
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings


class InteractivePlotter:
    """
    交互式图表生成器
    使用Plotly创建可交互的图表
    """
    
    def __init__(self, output_dir: str = "./results/figures"):
        self.output_dir = output_dir
        
    def plot_interactive_erp(
        self,
        evoked_data: Dict[str, np.ndarray],
        times: np.ndarray,
        channels: List[str],
        title: str = "ERP Waveforms"
    ) -> go.Figure:
        """
        绘制交互式ERP波形图
        
        Args:
            evoked_data: 各条件的ERP数据 {'condition': data}
            times: 时间点数组
            channels: 通道名称列表
            title: 图表标题
        
        Returns:
            Plotly图形对象
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for idx, (cond, data) in enumerate(evoked_data.items()):
            # 平均所有通道
            mean_data = np.mean(data, axis=0)
            std_data = np.std(data, axis=0)
            
            fig.add_trace(go.Scatter(
                x=times,
                y=mean_data,
                mode='lines',
                name=cond,
                line=dict(color=colors[idx % len(colors)], width=2),
                error_y=dict(
                    type='data',
                    array=std_data,
                    visible=True,
                    color=colors[idx % len(colors)]
                )
            ))
        
        # 添加垂直参考线
        fig.add_vline(x=0, line_dash="dash", line_color="gray",
                      annotation_text="Stimulus")
        
        # 添加水平参考线
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (µV)",
            hovermode='x unified',
            template='plotly_white',
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
        )
        
        return fig
    
    def plot_interactive_topomap(
        self,
        data: np.ndarray,
        channel_names: List[str],
        channel_positions: np.ndarray,
        time_points: List[float],
        title: str = "Topographic Maps"
    ) -> go.Figure:
        """
        绘制交互式脑地形图（可滑动时间轴）
        
        Args:
            data: 地形图数据 (n_channels, n_timepoints)
            channel_names: 通道名称列表
            channel_positions: 电极位置 (n_channels, 2)
            time_points: 时间点列表
            title: 图表标题
        
        Returns:
            Plotly图形对象
        """
        fig = go.Figure()
        
        # 创建滑动条
        steps = []
        for i, t in enumerate(time_points):
            # 创建插值网格（简化版）
            z_data = data[:, i].reshape(10, 10)  # 简化：需要实际插值
            
            step = dict(
                method="update",
                args=[{"z": [z_data]}],
                label=f"{t*1000:.0f} ms"
            )
            steps.append(step)
        
        # 初始显示第一个时间点
        fig.add_trace(go.Heatmap(
            z=data[:, 0].reshape(10, 10),
            colorscale='RdBu_r',
            showscale=True,
            colorbar=dict(title="µV")
        ))
        
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Time: "},
            pad={"t": 50},
            steps=steps
        )]
        
        fig.update_layout(
            title=title,
            sliders=sliders,
            template='plotly_white'
        )
        
        return fig
    
    def plot_interactive_spectrum(
        self,
        spectra: Dict[str, np.ndarray],
        frequencies: np.ndarray,
        title: str = "Power Spectrum"
    ) -> go.Figure:
        """
        绘制交互式频谱图
        
        Args:
            spectra: 各条件的频谱数据 {'condition': spectrum}
            frequencies: 频率数组
            title: 图表标题
        
        Returns:
            Plotly图形对象
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for idx, (cond, spectrum) in enumerate(spectra.items()):
            fig.add_trace(go.Scatter(
                x=frequencies,
                y=spectrum,
                mode='lines',
                name=cond,
                line=dict(color=colors[idx % len(colors)], width=2),
                fill='tozeroy' if idx == 0 else None
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power (dB)",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_interactive_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = "accuracy",
        title: str = "Algorithm Comparison"
    ) -> go.Figure:
        """
        绘制交互式算法对比图
        
        Args:
            results_df: 结果DataFrame (包含algorithm, metric, std等列)
            metric: 比较的指标
            title: 图表标题
        
        Returns:
            Plotly图形对象
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=results_df['algorithm'],
            y=results_df[metric],
            error_y=dict(
                type='data',
                array=results_df[f'{metric}_std'],
                visible=True
            ),
            marker_color=px.colors.qualitative.Set1,
            text=results_df[metric].round(3),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Algorithm",
            yaxis_title=metric.capitalize(),
            template='plotly_white',
            hovermode='closest'
        )
        
        return fig
    
    def plot_interactive_confusion_matrix(
        self,
        cm: np.ndarray,
        classes: List[str],
        title: str = "Confusion Matrix"
    ) -> go.Figure:
        """
        绘制交互式混淆矩阵
        
        Args:
            cm: 混淆矩阵
            classes: 类别名称列表
            title: 图表标题
        
        Returns:
            Plotly图形对象
        """
        # 归一化
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=classes,
            y=classes,
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            template='plotly_white'
        )
        
        return fig
    
    def plot_interactive_learning_curve(
        self,
        train_scores: List[float],
        val_scores: List[float],
        epochs: List[int],
        title: str = "Learning Curves"
    ) -> go.Figure:
        """
        绘制交互式学习曲线
        
        Args:
            train_scores: 训练集分数
            val_scores: 验证集分数
            epochs: 轮次列表
            title: 图表标题
        
        Returns:
            Plotly图形对象
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_scores,
            mode='lines+markers',
            name='Training',
            line=dict(color='#4C9F70', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_scores,
            mode='lines+markers',
            name='Validation',
            line=dict(color='#E9C46A', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_interactive_3d_scatter(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str] = None,
        title: str = "Feature Space Visualization"
    ) -> go.Figure:
        """
        绘制交互式3D特征空间散点图
        
        Args:
            features: 特征矩阵 (n_samples, n_features)
            labels: 标签
            feature_names: 特征名称
            title: 图表标题
        
        Returns:
            Plotly图形对象
        """
        # 使用PCA降维到3维
        from sklearn.decomposition import PCA
        
        if features.shape[1] > 3:
            pca = PCA(n_components=3)
            features_3d = pca.fit_transform(features)
            explained_var = pca.explained_variance_ratio_
            subtitle = f"PCA (explained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}, {explained_var[2]:.2f})"
        else:
            features_3d = features
            subtitle = ""
        
        fig = go.Figure()
        
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Set1
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            fig.add_trace(go.Scatter3d(
                x=features_3d[mask, 0],
                y=features_3d[mask, 1],
                z=features_3d[mask, 2],
                mode='markers',
                name=f'Class {label}',
                marker=dict(
                    size=5,
                    color=colors[i % len(colors)],
                    opacity=0.7
                )
            ))
        
        fig.update_layout(
            title=f"{title}<br><sup>{subtitle}</sup>",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3"
            ),
            template='plotly_white'
        )
        
        return fig