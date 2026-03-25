"""
可视化模块
提供论文级别图表和交互式可视化功能
"""

from .publication_plots import PublicationPlotter

# 尝试导入3D可视化，如果失败则跳过
try:
    from .brain_visualization import Brain3DVisualization
    BRAIN_VIZ_AVAILABLE = True
except ImportError:
    BRAIN_VIZ_AVAILABLE = False
    # 创建占位符
    class Brain3DVisualization:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyVista is not installed. Please install it with: pip install pyvista\n"
                "Or use 2D visualization methods instead."
            )

from .interactive_plots import InteractivePlotter

__all__ = [
    'PublicationPlotter',
    'Brain3DVisualization', 
    'InteractivePlotter'
]