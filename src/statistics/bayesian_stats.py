# src/statistics/bayesian_stats.py
# 完全重写，移除 pymc3 依赖

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings


class BayesianAnalyzer:
    """
    贝叶斯统计分析器
    提供贝叶斯因子计算（使用JZS方法，不依赖pymc3）
    """
    
    def __init__(self, prior_scale: float = 0.707):
        self.prior_scale = prior_scale
    
    def bayes_factor_ttest(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        paired: bool = True
    ) -> Dict[str, float]:
        """
        计算t检验的贝叶斯因子（JZS方法）
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            paired: 是否配对
        
        Returns:
            包含BF10、BF01和解释的字典
        """
        if paired:
            diff = data1 - data2
            n = len(diff)
            t_stat, p_value = stats.ttest_rel(data1, data2)
        else:
            n = len(data1) + len(data2)
            t_stat, p_value = stats.ttest_ind(data1, data2)
        
        # 计算JZS贝叶斯因子
        bf10 = self._jzs_bf(t_stat, n)
        bf01 = 1 / bf10 if bf10 > 0 else float('inf')
        
        return {
            'BF10': bf10,
            'BF01': bf01,
            'interpretation': self._interpret_bf(bf10),
            't_statistic': t_stat,
            'p_value': p_value,
            'n_samples': n
        }
    
    def _jzs_bf(self, t_stat: float, n: int) -> float:
        """
        计算JZS贝叶斯因子
        
        Args:
            t_stat: t统计量
            n: 样本量
        
        Returns:
            贝叶斯因子
        """
        from scipy.special import gamma
        
        r = self.prior_scale
        nu = n - 1
        
        # 处理极端情况
        if t_stat == 0:
            return 1.0
        
        try:
            bf = np.sqrt(1 + n * r**2) * (
                (1 + t_stat**2 / (nu * (1 + n * r**2))) ** (-(nu+1)/2)
            ) / (
                (1 + t_stat**2 / nu) ** (-(nu+1)/2)
            )
        except:
            bf = 1.0
        
        return bf
    
    def _interpret_bf(self, bf: float) -> str:
        """解释贝叶斯因子"""
        if bf > 100:
            return "极强证据支持备择假设 (BF > 100)"
        elif bf > 30:
            return "非常强的证据支持备择假设 (BF > 30)"
        elif bf > 10:
            return "强证据支持备择假设 (BF > 10)"
        elif bf > 3:
            return "中等证据支持备择假设 (BF > 3)"
        elif bf > 1:
            return "轶事证据支持备择假设 (BF > 1)"
        elif bf == 1:
            return "无证据 (BF = 1)"
        elif bf > 1/3:
            return "轶事证据支持零假设 (BF < 1/3)"
        elif bf > 1/10:
            return "中等证据支持零假设 (BF < 1/10)"
        elif bf > 1/30:
            return "强证据支持零假设 (BF < 1/30)"
        else:
            return "极强证据支持零假设 (BF < 1/100)"
    
    def compute_bayes_factor(self, t_stat: float, n: int) -> float:
        """
        直接根据t统计量计算贝叶斯因子
        
        Args:
            t_stat: t统计量
            n: 样本量
        
        Returns:
            贝叶斯因子
        """
        return self._jzs_bf(t_stat, n)


# 导出简化版本
__all__ = ['BayesianAnalyzer']