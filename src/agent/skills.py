from __future__ import annotations

from typing import Any, Dict, List

from .service import BCIAgentService


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_algorithm",
            "description": "设置BCI平台算法，比如用户说'用EEGNet算法'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "description": "算法名称，如 eegnet/svm/logistic_reg",
                    }
                },
                "required": ["algorithm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_mode",
            "description": "设置运行模式，single或benchmark。",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["single", "benchmark"],
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_preprocess",
            "description": "设置带通滤波参数，比如8-30Hz。",
            "parameters": {
                "type": "object",
                "properties": {
                    "low": {"type": "integer", "description": "低频截止"},
                    "high": {"type": "integer", "description": "高频截止"},
                },
                "required": ["low", "high"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_step",
            "description": "更新当前教学流程步骤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "step": {
                        "type": "string",
                        "enum": ["EEG采集", "预处理", "特征提取", "分类", "反馈"],
                    }
                },
                "required": ["step"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pipeline",
            "description": "运行完整或指定步骤的BCI流程。",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["EEG采集", "预处理", "特征提取", "分类", "反馈"],
                        },
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "根据运行结果生成图表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "description": "图表类型，可选时域/频域/地形图（也支持accuracy/time/frequency/topomap）",
                    },
                    "as_base64": {
                        "type": "boolean",
                        "description": "是否返回base64",
                        "default": False,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "根据运行结果生成报告。",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_format": {
                        "type": "string",
                        "enum": ["markdown", "json", "word"],
                        "default": "markdown",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json", "word"],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": "读取当前平台状态。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def execute_tool(service: BCIAgentService, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "set_algorithm":
        return service.set_algorithm(tool_args["algorithm"])
    if tool_name == "set_mode":
        return service.set_mode(tool_args["mode"])
    if tool_name == "set_preprocess":
        return service.set_preprocess(tool_args["low"], tool_args["high"])
    if tool_name == "set_step":
        return service.set_step(tool_args["step"])
    if tool_name == "run_pipeline":
        return service.run_pipeline(tool_args.get("steps"))
    if tool_name == "generate_chart":
        return service.generate_chart(
            chart_type=tool_args.get("chart_type", "accuracy"),
            as_base64=tool_args.get("as_base64", False),
        )
    if tool_name == "generate_report":
        report_format = tool_args.get("report_format") or tool_args.get("format") or "markdown"
        return service.generate_report(report_format)
    if tool_name == "get_state":
        return {"status": "success", "state": service.get_state()}
    return {"status": "error", "msg": f"未知工具: {tool_name}"}
