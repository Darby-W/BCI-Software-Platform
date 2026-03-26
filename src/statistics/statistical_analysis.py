"""
统计分析核心模块
将之前的 StatisticalAnalyzer 类完整放入此文件
"""

import numpy as np
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, mannwhitneyu, f_oneway
from sklearn.metrics import cohen_kappa_score
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class StatisticalTestResult:
    """统计检验结果"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    comparison_groups: List[str]

class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = []
    
    # ... 将之前的所有方法放入此处 ...
    # paired_ttest, independent_ttest, wilcoxon_test,
    # mann_whitney_test, anova_oneway, cross_validation_compare,
    # compute_agreement, power_analysis, compute_bayes_factor, format_results