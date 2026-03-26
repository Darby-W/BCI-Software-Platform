"""
报告生成器测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.reporting.report_generator import (
    ExperimentReportGenerator, ExperimentMetadata, SubjectInfo,
    DataAcquisition, PreprocessingParams, FeatureExtractionParams,
    ClassificationResults
)
from src.statistics.statistical_analysis import StatisticalTestResult


class TestReportGenerator:
    """报告生成器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.generator = ExperimentReportGenerator(output_dir="./test_reports")
        
        # 创建测试数据
        self.metadata = ExperimentMetadata(
            experiment_id="TEST_001",
            experiment_name="Test Experiment",
            experiment_date="2024-03-01",
            researcher="Test User",
            institution="Test University",
            description="Test description"
        )
        
        self.subject_info = SubjectInfo(
            subject_id="S001",
            age=25,
            gender="M",
            handedness="R",
            experience_level="naive",
            inclusion_criteria=["Healthy"]
        )
        
        self.acquisition = DataAcquisition(
            device_name="Test Device",
            channel_count=32,
            sampling_rate=500,
            channel_names=["Fz", "C3", "Cz", "C4", "Pz"],
            reference="FCz",
            ground="AFz",
            impedance_threshold=10
        )
        
        self.preprocessing = PreprocessingParams(
            bandpass_low=0.5,
            bandpass_high=40,
            notch_frequency=50,
            artifact_removal_method="ICA",
            eog_channels=["Fp1", "Fp2"],
            bad_channels=[]
        )
        
        self.feature_params = FeatureExtractionParams(
            method="CSP",
            time_window=[0.5, 2.5],
            frequency_bands=[{"name": "mu", "range": [8, 12]}],
            n_components=4
        )
        
        self.results = ClassificationResults(
            algorithm="EEGNet",
            accuracy=0.85,
            accuracy_std=0.02,
            f1_score=0.84,
            f1_std=0.02,
            kappa=0.70,
            kappa_std=0.05,
            auc=0.92,
            auc_std=0.03,
            confusion_matrix=np.array([[85, 15], [12, 88]]),
            training_time=45.2,
            inference_time=2.3,
            cv_method="10-fold CV",
            n_folds=10
        )
    
    def test_generate_report(self):
        """测试报告生成"""
        report_path = self.generator.generate_comprehensive_report(
            metadata=self.metadata,
            subject_info=self.subject_info,
            acquisition=self.acquisition,
            preprocessing=self.preprocessing,
            feature_params=self.feature_params,
            results=self.results,
            statistical_tests=[],
            figures={},
            additional_notes="Test notes"
        )
        
        assert report_path is not None
        assert os.path.exists(report_path)
        assert self.metadata.experiment_id in report_path
    
    def test_markdown_export(self):
        """测试Markdown导出"""
        md_content = self.generator._generate_markdown_report(
            metadata=self.metadata,
            subject_info=self.subject_info,
            acquisition=self.acquisition,
            preprocessing=self.preprocessing,
            feature_params=self.feature_params,
            results=self.results,
            statistical_tests=[],
            additional_notes=""
        )
        
        assert "Test Experiment" in md_content
        assert "EEGNet" in md_content
        assert "0.85" in md_content


if __name__ == "__main__":
    pytest.main([__file__])