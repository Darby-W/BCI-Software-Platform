"""
实验报告自动生成器
将之前的 ExperimentReportGenerator 类完整放入此文件
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from pathlib import Path

# 导入其他模块
from src.statistics.statistical_analysis import StatisticalAnalyzer, StatisticalTestResult
from src.visualization.publication_plots import PublicationPlotter

@dataclass
class ExperimentMetadata:
    """实验元数据"""
    experiment_id: str
    experiment_name: str
    experiment_date: str
    researcher: str
    institution: str
    description: str

# ... 其他数据类定义 ...

class ExperimentReportGenerator:
    """实验报告生成器"""
    
    def __init__(self, output_dir: str = None):
        project_root = Path(__file__).resolve().parents[2]
        default_output = project_root / "results" / "reports"
        self.output_dir = Path(output_dir) if output_dir else default_output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stat_analyzer = StatisticalAnalyzer()
        self.plotter = PublicationPlotter()
    
    # ... 将之前的所有方法放入此处 ...