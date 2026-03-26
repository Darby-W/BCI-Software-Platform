"""
报告导出模块
支持多种格式导出
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False


class ReportExporter:
    """
    报告导出器
    支持导出为PDF、DOCX、JSON等格式
    """
    
    def __init__(self, output_dir: str = "./results/exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_pdf(
        self,
        html_content: str,
        filename: str,
        title: str = "BCI Experiment Report"
    ) -> Optional[str]:
        """
        导出为PDF格式
        
        Args:
            html_content: HTML内容
            filename: 输出文件名
            title: 文档标题
        
        Returns:
            输出文件路径
        """
        if not WEASYPRINT_AVAILABLE:
            warnings.warn("WeasyPrint not installed. PDF export disabled.")
            return None
        
        output_path = self.output_dir / f"{filename}.pdf"
        
        try:
            HTML(string=html_content).write_pdf(output_path)
            return str(output_path)
        except Exception as e:
            print(f"PDF export failed: {e}")
            return None
    
    def export_to_json(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> str:
        """
        导出为JSON格式
        
        Args:
            data: 要导出的数据
            filename: 输出文件名
        
        Returns:
            输出文件路径
        """
        output_path = self.output_dir / f"{filename}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def export_to_csv(
        self,
        data: pd.DataFrame,
        filename: str
    ) -> str:
        """
        导出为CSV格式
        
        Args:
            data: 要导出的DataFrame
            filename: 输出文件名
        
        Returns:
            输出文件路径
        """
        output_path = self.output_dir / f"{filename}.csv"
        data.to_csv(output_path, index=False, encoding='utf-8-sig')
        return str(output_path)
    
    def export_to_excel(
        self,
        data_dict: Dict[str, pd.DataFrame],
        filename: str
    ) -> str:
        """
        导出为Excel格式（多工作表）
        
        Args:
            data_dict: 工作表名到DataFrame的映射
            filename: 输出文件名
        
        Returns:
            输出文件路径
        """
        output_path = self.output_dir / f"{filename}.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return str(output_path)
    
    def export_to_markdown(
        self,
        content: str,
        filename: str
    ) -> str:
        """
        导出为Markdown格式
        
        Args:
            content: Markdown内容
            filename: 输出文件名
        
        Returns:
            输出文件路径
        """
        output_path = self.output_dir / f"{filename}.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_path)