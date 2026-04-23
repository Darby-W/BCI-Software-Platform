from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os

from .service import BCIAgentService
from .skills import TOOL_DEFINITIONS, execute_tool


SYSTEM_PROMPT = (
    "你是BCI康复训练平台智能助手。你会先理解用户意图，再按需调用工具完成参数设置、算法运行、"
    "图表与报告生成。你必须尽量自动完成流程并返回最终可用结果路径。"
)


class OpenAIBCIAgent:
    def __init__(
        self,
        service: BCIAgentService,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.service = service
        self.model = model

        if client is not None:
            self.client = client
            return

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("未安装 openai SDK，请先安装: pip install openai") from exc

        resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("未检测到 API Key，请设置 DEEPSEEK_API_KEY（或 OPENAI_API_KEY）")

        resolved_base_url = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.deepseek.com"
        )

        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        self.api_key_last4 = resolved_api_key[-4:]
        self.base_url = resolved_base_url
        self._session_messages: Dict[str, List[Dict[str, Any]]] = {}

    def chat(
        self,
        user_input: str,
        session_id: str = "default",
        reset: bool = False,
        max_rounds: int = 10,
    ) -> Dict[str, Any]:
        if reset or session_id not in self._session_messages:
            self._session_messages[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

        messages = self._session_messages[session_id]
        messages.append({"role": "user", "content": user_input})

        used_tools: List[str] = []
        artifacts: List[Dict[str, Any]] = []
        tool_signature_counts: Dict[str, int] = {}
        last_tool_result: Optional[Dict[str, Any]] = None

        for _ in range(max_rounds):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            messages.append(msg)

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                # 截断历史，防止上下文无限增长
                if len(messages) > 60:
                    self._session_messages[session_id] = [messages[0]] + messages[-59:]
                return {
                    "status": "success",
                    "reply": msg.content,
                    "used_tools": used_tools,
                    "session_id": session_id,
                    "artifacts": artifacts,
                }

            for call in tool_calls:
                tool_name = call.function.name
                try:
                    tool_args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_args = {}

                tool_result = execute_tool(self.service, tool_name, tool_args)
                last_tool_result = tool_result if isinstance(tool_result, dict) else None
                used_tools.append(tool_name)
                self._collect_artifacts(tool_name, tool_result, artifacts)

                signature = f"{tool_name}:{json.dumps(tool_args, ensure_ascii=False, sort_keys=True)}"
                tool_signature_counts[signature] = tool_signature_counts.get(signature, 0) + 1
                if tool_signature_counts[signature] >= 3:
                    reply = self._build_fallback_reply(used_tools, artifacts, last_tool_result)
                    return {
                        "status": "success",
                        "reply": reply,
                        "used_tools": used_tools,
                        "session_id": session_id,
                        "artifacts": artifacts,
                    }

                if tool_name == "run_pipeline" and isinstance(tool_result, dict):
                    if tool_result.get("status") == "error":
                        err_msg = tool_result.get("msg") or "Pipeline 执行失败"
                        return {
                            "status": "error",
                            "msg": err_msg,
                            "reply": err_msg,
                            "used_tools": used_tools,
                            "session_id": session_id,
                            "artifacts": artifacts,
                            "hint": tool_result.get("hint"),
                        }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": tool_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        if artifacts or used_tools:
            # 工具轮次耗尽时，尝试再做一次“无工具总结”生成，避免直接报错给用户。
            try:
                final_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                final_msg = final_resp.choices[0].message.content
                reply = final_msg or self._build_fallback_reply(used_tools, artifacts, last_tool_result)
            except Exception:
                reply = self._build_fallback_reply(used_tools, artifacts, last_tool_result)

            return {
                "status": "success",
                "reply": reply,
                "used_tools": used_tools,
                "session_id": session_id,
                "artifacts": artifacts,
            }

        return {
            "status": "error",
            "reply": "工具调用轮次超过上限，请简化需求后重试。",
            "used_tools": used_tools,
            "session_id": session_id,
            "artifacts": artifacts,
        }

    def reset_session(self, session_id: str = "default") -> Dict[str, Any]:
        self._session_messages[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        return {"status": "success", "session_id": session_id}

    def _collect_artifacts(
        self,
        tool_name: str,
        tool_result: Dict[str, Any],
        artifacts: List[Dict[str, Any]],
    ) -> None:
        if not isinstance(tool_result, dict):
            return

        if tool_name == "generate_chart" and tool_result.get("status") == "success":
            chart_path = tool_result.get("chart_path")
            if chart_path:
                artifacts.append(
                    {
                        "type": "chart",
                        "path": chart_path,
                        "chart_type": tool_result.get("chart_type", "unknown"),
                    }
                )

        if tool_name == "generate_report":
            report_path = tool_result.get("report_path")
            if report_path:
                artifacts.append(
                    {
                        "type": "report",
                        "path": report_path,
                        "format": tool_result.get("format", "unknown"),
                    }
                )

            fallback_markdown = tool_result.get("fallback_markdown")
            if fallback_markdown:
                artifacts.append(
                    {
                        "type": "report",
                        "path": fallback_markdown,
                        "format": "markdown",
                    }
                )

    def _build_fallback_reply(
        self,
        used_tools: List[str],
        artifacts: List[Dict[str, Any]],
        last_tool_result: Optional[Dict[str, Any]],
    ) -> str:
        lines = ["已完成工具执行，结果如下："]
        if used_tools:
            lines.append(f"- 调用工具: {', '.join(used_tools)}")

        chart_paths = [a.get("path") for a in artifacts if a.get("type") == "chart" and a.get("path")]
        report_paths = [a.get("path") for a in artifacts if a.get("type") == "report" and a.get("path")]

        if chart_paths:
            lines.append(f"- 已生成图表: {chart_paths[-1]}")
        if report_paths:
            lines.append(f"- 已生成报告: {report_paths[-1]}")

        if last_tool_result and isinstance(last_tool_result, dict):
            msg = last_tool_result.get("msg")
            if msg:
                lines.append(f"- 说明: {msg}")

        if not chart_paths and not report_paths:
            lines.append("- 未产出图表/报告，请检查数据与依赖后重试。")
        return "\n".join(lines)
