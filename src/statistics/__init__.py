"""
统计分析模块
提供BCI实验常用的统计检验方法
"""

from .statistical_analysis import StatisticalAnalyzer, StatisticalTestResult
from .power_analysis import PowerAnalyzer
from .bayesian_stats import BayesianAnalyzer

__all__ = [
    'StatisticalAnalyzer',
    'StatisticalTestResult', 
    'PowerAnalyzer',
    'BayesianAnalyzer'
]