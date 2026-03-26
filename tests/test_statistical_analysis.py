"""
统计分析模块测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.statistics.statistical_analysis import StatisticalAnalyzer, StatisticalTestResult
from src.statistics.power_analysis import PowerAnalyzer
from src.statistics.bayesian_stats import BayesianAnalyzer


class TestStatisticalAnalyzer:
    """统计分析器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = StatisticalAnalyzer(alpha=0.05)
        
        # 生成测试数据
        np.random.seed(42)
        self.group1 = np.random.normal(0.8, 0.1, 30)  # 准确率 ~80%
        self.group2 = np.random.normal(0.75, 0.12, 30)  # 准确率 ~75%
        self.paired_before = np.random.normal(0.7, 0.1, 25)
        self.paired_after = self.paired_before + np.random.normal(0.05, 0.05, 25)
    
    def test_paired_ttest(self):
        """测试配对t检验"""
        result = self.analyzer.paired_ttest(
            self.paired_before,
            self.paired_after,
            ('Before', 'After')
        )
        
        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == "Paired t-test"
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.significant, bool)
        assert isinstance(result.effect_size, float)
        assert len(result.confidence_interval) == 2
        assert result.comparison_groups == ['Before', 'After']
        
        # 验证统计量计算正确
        diff = self.paired_after - self.paired_before
        expected_t_stat = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
        assert abs(result.statistic - expected_t_stat) < 0.01
    
    def test_independent_ttest(self):
        """测试独立样本t检验"""
        result = self.analyzer.independent_ttest(
            self.group1,
            self.group2,
            ('Group1', 'Group2')
        )
        
        assert result.test_name == "Independent t-test"
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        
        # 验证效应量计算
        assert 0 <= result.effect_size <= 2
    
    def test_wilcoxon_test(self):
        """测试Wilcoxon符号秩检验"""
        result = self.analyzer.wilcoxon_test(
            self.paired_before,
            self.paired_after,
            ('Before', 'After')
        )
        
        assert result.test_name == "Wilcoxon Signed-Rank Test"
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        
        # 验证非参数检验结果
        assert result.p_value >= 0
    
    def test_mann_whitney_test(self):
        """测试Mann-Whitney U检验"""
        result = self.analyzer.mann_whitney_test(
            self.group1,
            self.group2,
            ('Group1', 'Group2')
        )
        
        assert result.test_name == "Mann-Whitney U Test"
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
    
    def test_anova_oneway(self):
        """测试单因素方差分析"""
        # 创建三组数据
        group_a = np.random.normal(0.8, 0.1, 20)
        group_b = np.random.normal(0.82, 0.1, 20)
        group_c = np.random.normal(0.75, 0.1, 20)
        
        result = self.analyzer.anova_oneway(
            group_a, group_b, group_c,
            group_names=['A', 'B', 'C']
        )
        
        assert result.test_name == "One-way ANOVA"
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.effect_size, float)
        assert 0 <= result.effect_size <= 1
    
    def test_cross_validation_compare(self):
        """测试交叉验证比较"""
        scores_dict = {
            'Algorithm_A': np.random.normal(0.85, 0.02, 10),
            'Algorithm_B': np.random.normal(0.82, 0.03, 10),
            'Algorithm_C': np.random.normal(0.87, 0.02, 10)
        }
        
        results = self.analyzer.cross_validation_compare(scores_dict, test_method='wilcoxon')
        
        assert len(results) == 3  # C(3,2) = 3
        assert ('Algorithm_A', 'Algorithm_B') in results
        assert ('Algorithm_A', 'Algorithm_C') in results
        assert ('Algorithm_B', 'Algorithm_C') in results
        
        for key, result in results.items():
            assert isinstance(result, StatisticalTestResult)
    
    def test_compute_agreement(self):
        """测试一致性计算"""
        scores_1 = np.array([1, 2, 3, 4, 5])
        scores_2 = np.array([1, 2, 3, 4, 6])  # 高度相关
        
        agreement = self.analyzer.compute_agreement(scores_1, scores_2)
        
        assert 'kappa' in agreement
        assert 'pearson_r' in agreement
        assert 'spearman_r' in agreement
        assert agreement['pearson_r'] > 0.9
    
    def test_format_results(self):
        """测试结果格式化"""
        result = self.analyzer.paired_ttest(
            self.paired_before,
            self.paired_after,
            ('Before', 'After')
        )
        
        formatted = self.analyzer.format_results([result])
        
        assert "Paired t-test" in formatted
        assert "统计量" in formatted
        assert "p值" in formatted
        assert "效应量" in formatted


class TestPowerAnalyzer:
    """功效分析器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = PowerAnalyzer(alpha=0.05, power=0.8)
    
    def test_sample_size_for_ttest(self):
        """测试t检验样本量计算"""
        # 大效应量
        n_large = self.analyzer.sample_size_for_ttest(0.8, 'paired')
        # 中效应量
        n_medium = self.analyzer.sample_size_for_ttest(0.5, 'paired')
        # 小效应量
        n_small = self.analyzer.sample_size_for_ttest(0.2, 'paired')
        
        # 效应量越小，所需样本量越大
        assert n_large < n_medium < n_small
        assert isinstance(n_large, int)
    
    def test_achieved_power(self):
        """测试统计功效计算"""
        power = self.analyzer.achieved_power(0.5, 50, 'paired')
        
        assert 0 <= power <= 1
        assert isinstance(power, float)
    
    def test_effect_size_interpretation(self):
        """测试效应量解释"""
        # 大效应
        large_effect = self.analyzer.effect_size_interpretation(0.9)
        assert "大效应" in large_effect
        
        # 中等效应
        medium_effect = self.analyzer.effect_size_interpretation(0.6)
        assert "中等效应" in medium_effect
        
        # 小效应
        small_effect = self.analyzer.effect_size_interpretation(0.3)
        assert "小效应" in small_effect
    
    def test_generate_power_report(self):
        """测试功效报告生成"""
        report = self.analyzer.generate_power_report(0.5, 30, 'paired')
        
        assert "功效分析报告" in report
        assert "样本量" in report
        assert "统计功效" in report


class TestBayesianAnalyzer:
    """贝叶斯分析器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = BayesianAnalyzer(prior_scale=0.707)
        
        np.random.seed(42)
        self.data1 = np.random.normal(0.8, 0.1, 30)
        self.data2 = np.random.normal(0.75, 0.12, 30)
        self.paired_data = np.random.normal(0.05, 0.1, 25)
    
    def test_bayes_factor_ttest_paired(self):
        """测试配对t检验的贝叶斯因子"""
        result = self.analyzer.bayes_factor_ttest(self.data1, self.data2, paired=True)
        
        assert 'BF10' in result
        assert 'BF01' in result
        assert 'interpretation' in result
        assert result['BF10'] > 0
        assert result['BF01'] > 0
        
        # BF10和BF01互为倒数
        assert abs(result['BF10'] * result['BF01'] - 1) < 1e-6
    
    def test_bayes_factor_ttest_independent(self):
        """测试独立样本t检验的贝叶斯因子"""
        result = self.analyzer.bayes_factor_ttest(self.data1, self.data2, paired=False)
        
        assert 'BF10' in result
        assert 'BF01' in result
    
    def test_bayes_factor_proportion(self):
        """测试比例检验的贝叶斯因子"""
        # 30次试验，20次成功
        result = self.analyzer.bayes_factor_proportion(
            successes=20,
            trials=30,
            null_proportion=0.5
        )
        
        assert 'BF10' in result
        assert 'BF01' in result
        assert result['BF10'] > 0
    
    def test_interpret_bf(self):
        """测试贝叶斯因子解释"""
        # 强证据
        strong = self.analyzer._interpret_bf(15)
        assert "强证据" in strong
        
        # 中等证据
        medium = self.analyzer._interpret_bf(5)
        assert "中等证据" in medium
        
        # 轶事证据
        anecdotal = self.analyzer._interpret_bf(2)
        assert "轶事证据" in anecdotal
        
        # 等于1
        equal = self.analyzer._interpret_bf(1)
        assert "无证据" in equal


if __name__ == "__main__":
    pytest.main([__file__])