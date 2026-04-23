"""
论文级别图表生成器
将之前的 PublicationPlotter 类完整放入此文件
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path

# 设置论文级图表样式
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)

# 专业配色方案
COLORS = {
    'left': '#2E86AB',
    'right': '#A23B72',
    'baseline': '#F18F01',
    'significant': '#73AB84',
    'non_significant': '#C73E1D',
    'train': '#4C9F70',
    'test': '#E9C46A'
}

class PublicationPlotter:
    """
    论文级图表生成器
    """
    
    def __init__(self, output_dir: str = None):
        project_root = Path(__file__).resolve().parents[2]
        default_output = project_root / "results" / "figures"
        self.output_dir = Path(output_dir) if output_dir else default_output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures = {}
    
    # ... 将之前的所有方法放入此处 ...
    # plot_confusion_matrix, plot_roc_curve, plot_erp_waveforms,
    # plot_topomap, plot_accuracy_comparison, plot_time_frequency,
    # plot_feature_importance, plot_learning_curves