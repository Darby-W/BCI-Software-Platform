"""
报告生成模块
自动生成实验报告
"""

from .report_generator import ExperimentReportGenerator
from .export_formats import ReportExporter

__all__ = [
    'ExperimentReportGenerator',
    'ReportExporter'
]