from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import base64
import importlib.util
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .openai_agent import OpenAIBCIAgent
from .service import BCIAgentService
from .skills import TOOL_DEFINITIONS


class AlgorithmRequest(BaseModel):
    algorithm: str


class ModeRequest(BaseModel):
    mode: str = Field(pattern="^(single|benchmark)$")


class PreprocessRequest(BaseModel):
    low: int
    high: int


class StepRequest(BaseModel):
    step: str


class RunPipelineRequest(BaseModel):
    steps: Optional[List[str]] = None


class ChatRequest(BaseModel):
    user_input: str
    model: Optional[str] = None
    session_id: str = "streamlit-default"
    reset: bool = False


class LLMConfigRequest(BaseModel):
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"


def get_default_model() -> str:
    # DeepSeek OpenAI-compatible default model.
    return os.getenv("AGENT_LLM_MODEL", "deepseek-chat")


class UploadDataRequest(BaseModel):
    file_name: str
    content_base64: str


def create_app() -> FastAPI:
    app = FastAPI(title="BCI Agent API", version="0.1.0")
    service = BCIAgentService()
    app.state.agent_service = service
    app.state.agent_client = None
    app.state.llm_config = {
        "api_key": os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
        "base_url": os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com",
        "model": get_default_model(),
    }

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok"}

    @app.get("/api/skills")
    def get_skills() -> List[Dict[str, Any]]:
        return TOOL_DEFINITIONS

    @app.get("/api/state")
    def get_state() -> Dict[str, Any]:
        return {"status": "success", "state": service.get_state()}

    @app.get("/api/llm-config")
    def get_llm_config() -> Dict[str, Any]:
        cfg = app.state.llm_config
        key = cfg.get("api_key", "")
        masked = "" if not key else ("*" * max(len(key) - 4, 0) + key[-4:])
        return {
            "status": "success",
            "config": {
                "api_key_masked": masked,
                "has_api_key": bool(key),
                "base_url": cfg.get("base_url", "https://api.deepseek.com"),
                "model": cfg.get("model", get_default_model()),
            },
        }

    @app.post("/api/llm-config")
    def set_llm_config(req: LLMConfigRequest) -> Dict[str, Any]:
        existing_key = app.state.llm_config.get("api_key", "")
        new_key = req.api_key.strip()
        if new_key == "__KEEP_EXISTING__":
            new_key = existing_key

        app.state.llm_config = {
            "api_key": new_key,
            "base_url": req.base_url.strip() or "https://api.deepseek.com",
            "model": req.model.strip() or "deepseek-chat",
        }
        # 强制重建，确保后续对话使用新配置。
        app.state.agent_client = None
        return {
            "status": "success",
            "msg": "LLM 配置已更新",
        }

    @app.post("/api/set-algorithm")
    def set_algorithm(req: AlgorithmRequest) -> Dict[str, Any]:
        return service.set_algorithm(req.algorithm)

    @app.post("/api/set-mode")
    def set_mode(req: ModeRequest) -> Dict[str, Any]:
        return service.set_mode(req.mode)

    @app.post("/api/set-preprocess")
    def set_preprocess(req: PreprocessRequest) -> Dict[str, Any]:
        return service.set_preprocess(req.low, req.high)

    multipart_available = importlib.util.find_spec("multipart") is not None

    if multipart_available:
        @app.post("/api/upload-data")
        async def upload_data(file: UploadFile = File(...)) -> Dict[str, Any]:
            content = await file.read()
            return service.upload_data(filename=file.filename, file_bytes=content)
    else:
        @app.post("/api/upload-data")
        def upload_data_json(req: UploadDataRequest) -> Dict[str, Any]:
            try:
                content = base64.b64decode(req.content_base64)
            except Exception as exc:
                return {
                    "status": "error",
                    "msg": f"base64 解码失败: {exc}",
                    "hint": "建议安装 python-multipart 以支持表单文件上传。",
                }
            result = service.upload_data(filename=req.file_name, file_bytes=content)
            result["hint"] = "当前为 JSON/base64 上传模式（因缺少 python-multipart）。"
            return result

    @app.post("/api/set-step")
    def set_step(req: StepRequest) -> Dict[str, Any]:
        return service.set_step(req.step)

    @app.post("/api/run-pipeline")
    def run_pipeline_api(req: RunPipelineRequest) -> Dict[str, Any]:
        return service.run_pipeline(steps=req.steps)

    @app.get("/api/task/{task_id}")
    def get_task(task_id: str) -> Dict[str, Any]:
        return service.get_task(task_id)

    @app.get("/api/generate-chart")
    def generate_chart(chart_type: str = "accuracy", as_base64: bool = False) -> Dict[str, Any]:
        return service.generate_chart(chart_type=chart_type, as_base64=as_base64)

    @app.get("/api/generate-report")
    def generate_report(report_format: str = "markdown", format: Optional[str] = None) -> Dict[str, Any]:
        target_format = format or report_format
        return service.generate_report(report_format=target_format)

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> Dict[str, Any]:
        if app.state.agent_client is None:
            try:
                cfg = app.state.llm_config
                app.state.agent_client = OpenAIBCIAgent(
                    service=service,
                    model=req.model or cfg.get("model", get_default_model()),
                    api_key=cfg.get("api_key") or None,
                    base_url=cfg.get("base_url") or None,
                )
            except Exception as exc:
                return {
                    "status": "error",
                    "msg": str(exc),
                    "hint": "请设置 DEEPSEEK_API_KEY（或 OPENAI_API_KEY）并安装 openai 依赖。",
                }
        if req.model and req.model != app.state.agent_client.model:
            app.state.agent_client.model = req.model
        return app.state.agent_client.chat(
            req.user_input,
            session_id=req.session_id,
            reset=req.reset,
        )

    @app.post("/api/chat/reset")
    def reset_chat(session_id: str = "streamlit-default") -> Dict[str, Any]:
        if app.state.agent_client is None:
            try:
                app.state.agent_client = OpenAIBCIAgent(service=service)
            except Exception as exc:
                return {"status": "error", "msg": str(exc)}
        return app.state.agent_client.reset_session(session_id=session_id)

    project_root = Path(__file__).resolve().parents[2]
    app.mount("/results", StaticFiles(directory=str(project_root / "results")), name="results")
    return app


app = create_app()
