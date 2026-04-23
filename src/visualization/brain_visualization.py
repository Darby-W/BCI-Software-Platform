"""
3D大脑动态可视化模块
使用PyVista/Three.js实现大脑3D可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import warnings
from pathlib import Path

# 尝试导入PyVista，如果失败则创建占位符
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    # 创建一个占位符类，避免 NameError
    class _Placeholder:
        def __getattr__(self, name):
            return None
    pv = _Placeholder()
    warnings.warn("pyvista not installed. 3D visualization will be disabled.")


class Brain3DVisualization:
    """
    3D大脑动态可视化
    支持脑区激活、源定位、脑网络可视化
    """
    
    def __init__(self, output_dir: str = None):
        project_root = Path(__file__).resolve().parents[2]
        default_output = project_root / "results" / "figures"
        self.output_dir = Path(output_dir) if output_dir else default_output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plotter = None
        self._pyvista_available = PYVISTA_AVAILABLE
        
    def _check_pyvista(self):
        """检查PyVista是否可用"""
        if not self._pyvista_available:
            raise ImportError(
                "PyVista is not installed. Please install it with: pip install pyvista\n"
                "Or use 2D visualization methods instead."
            )
    
    def create_brain_mesh(
        self,
        template: str = "fsaverage",
        hemi: str = "both"
    ):
        """
        创建大脑3D网格
        
        Args:
            template: 大脑模板 ('fsaverage', 'MNI152', 'ICBM152')
            hemi: 半球 ('left', 'right', 'both')
        
        Returns:
            PyVista网格对象，如果不可用则返回None
        """
        if not self._pyvista_available:
            print("PyVista not available. Skipping 3D visualization.")
            return None
        
        # 创建简化的大脑球体模型（示例）
        # 实际应使用mne或nibabel加载真实大脑模板
        sphere = pv.Sphere(radius=1.0, center=(0, 0, 0))
        
        return sphere
    
    def add_activation(
        self,
        brain_mesh,
        activation_data: np.ndarray,
        channels: List[str],
        cmap: str = "RdBu_r",
        opacity: float = 0.7
    ):
        """
        添加脑区激活效果
        
        Args:
            brain_mesh: 大脑网格
            activation_data: 激活数据 (n_channels,)
            channels: 通道名称列表
            cmap: 颜色映射
            opacity: 透明度
        
        Returns:
            PyVista绘图器，如果不可用则返回None
        """
        if not self._pyvista_available or brain_mesh is None:
            return None
        
        plotter = pv.Plotter()
        plotter.add_mesh(
            brain_mesh,
            cmap=cmap,
            opacity=opacity,
            show_scalar_bar=True
        )
        
        # 添加电极点（简化为随机位置）
        for i, ch in enumerate(channels[:10]):  # 限制数量
            pos = np.random.randn(3) * 0.8
            intensity = activation_data[i] if i < len(activation_data) else 0.5
            
            sphere = pv.Sphere(radius=0.05, center=pos)
            plotter.add_mesh(
                sphere,
                color=plt.cm.RdBu_r(intensity),
                show_scalar_bar=False
            )
        
        return plotter
    
    def visualize_source_localization(
        self,
        source_activity: np.ndarray,
        vertices: np.ndarray,
        time_point: float = 0
    ) -> plt.Figure:
        """
        可视化源定位结果（2D版本，不依赖PyVista）
        
        Args:
            source_activity: 源活动数据 (n_sources,)
            vertices: 源空间顶点坐标 (n_sources, 3)
            time_point: 时间点
        
        Returns:
            Matplotlib图形
        """
        fig = plt.figure(figsize=(12, 8))
        
        # 创建3D子图
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制源点
        scatter = ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            c=source_activity,
            cmap='hot',
            s=20,
            alpha=0.6
        )
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Source Localization at t = {time_point} s')
        
        plt.colorbar(scatter, label='Activity (µV)')
        
        plt.tight_layout()
        return fig
    
    def visualize_connectivity_network(
        self,
        connectivity_matrix: np.ndarray,
        channel_names: List[str],
        threshold: float = 0.5,
        layout: str = "circular"
    ) -> plt.Figure:
        """
        可视化脑网络连接
        
        Args:
            connectivity_matrix: 连接矩阵 (n_channels, n_channels)
            channel_names: 通道名称列表
            threshold: 连接阈值
            layout: 布局方式 ('circular', 'spring', 'shell')
        
        Returns:
            Matplotlib图形
        """
        try:
            import networkx as nx
        except ImportError:
            # 如果没有networkx，返回空图
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.text(0.5, 0.5, "NetworkX not installed.\nPlease install: pip install networkx",
                   ha='center', va='center', fontsize=12)
            return fig
        
        n_channels = len(channel_names)
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for i, name in enumerate(channel_names):
            G.add_node(i, label=name)
        
        # 添加边（超过阈值的连接）
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                weight = connectivity_matrix[i, j]
                if weight > threshold:
                    G.add_edge(i, j, weight=weight)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 选择布局
        if layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G, k=3, iterations=50)
        else:
            pos = nx.shell_layout(G)
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos, node_size=500, node_color='lightblue',
            edgecolors='blue', linewidths=2, ax=ax
        )
        
        # 绘制边（根据权重调整宽度）
        edges = G.edges()
        if edges:
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(
                G, pos, width=np.array(weights) * 5,
                edge_color='gray', alpha=0.6, ax=ax
            )
        
        # 绘制标签
        labels = {i: name for i, name in enumerate(channel_names)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Brain Connectivity Network (threshold={threshold})')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_animation(
        self,
        time_series: np.ndarray,
        vertices: np.ndarray,
        times: np.ndarray,
        output_file: str = "brain_animation.gif"
    ):
        """
        创建动态大脑动画（2D版本，不依赖PyVista）
        
        Args:
            time_series: 时间序列数据 (n_times, n_sources)
            vertices: 顶点坐标
            times: 时间点
            output_file: 输出文件名
        """
        import matplotlib.animation as animation
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 确定颜色范围
        vmin = np.min(time_series)
        vmax = np.max(time_series)
        
        def update_frame(frame):
            ax.clear()
            activity = time_series[frame]
            
            scatter = ax.scatter(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                c=activity, cmap='hot', s=20, alpha=0.7,
                vmin=vmin, vmax=vmax
            )
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title(f't = {times[frame]:.3f} s')
            return scatter,
        
        anim = animation.FuncAnimation(
            fig, update_frame, frames=len(times),
            interval=100, blit=False
        )
        
        # 确保输出目录存在
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_path = os.path.join(self.output_dir, output_file)
        anim.save(output_path, writer='pillow')
        plt.close()
        
        print(f"Animation saved to {output_path}")
    
    def plot_3d_brain_static(
        self,
        eeg_data: np.ndarray,
        channel_positions: np.ndarray,
        channel_names: List[str],
        title: str = "3D Brain Visualization"
    ) -> plt.Figure:
        """
        静态3D大脑可视化（使用matplotlib，不依赖PyVista）
        
        Args:
            eeg_data: EEG数据 (n_channels, n_times)
            channel_positions: 电极位置 (n_channels, 3)
            channel_names: 通道名称列表
            title: 图表标题
        
        Returns:
            Matplotlib图形
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 计算每个通道的平均功率
        power = np.mean(eeg_data ** 2, axis=1)
        power_norm = (power - power.min()) / (power.max() - power.min())
        
        # 绘制电极点
        scatter = ax.scatter(
            channel_positions[:, 0],
            channel_positions[:, 1],
            channel_positions[:, 2],
            c=power_norm,
            cmap='RdBu_r',
            s=100 * (power_norm + 0.5),
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5
        )
        
        # 添加电极标签
        for i, name in enumerate(channel_names):
            ax.text(
                channel_positions[i, 0],
                channel_positions[i, 1],
                channel_positions[i, 2],
                name,
                fontsize=8,
                ha='center',
                va='bottom'
            )
        
        # 绘制大脑轮廓（简化的球体）
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 0.9 * np.outer(np.cos(u), np.sin(v))
        y = 0.9 * np.outer(np.sin(u), np.sin(v))
        z = 0.9 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, color='gray')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        plt.colorbar(scatter, label='Normalized Power')
        
        plt.tight_layout()
        return fig