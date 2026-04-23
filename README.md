BCI-Software-Platform

A modular Brain-Computer Interface (BCI) software platform for offline experiment management, data processing, and algorithm evaluation.

This project is developed as a research-oriented platform with a clear separation of:

- Data Management
- Algorithm Module (Plugin-based)
- Experiment Pipeline
- Reproducible Results

---

1.Clone the Repository
git clone https://github.com/Darby-W/BCI-Software-Platform.git
cd BCI-Software-Platform

---

2.Install Dependencies
We recommend using a virtual environment (optional but recommended):
pip install -r requirements.txt
Currently required:numpy

---

3.Run the MVP
python run_mvp.py
If successful, you should see:
[OK] run_id=...
[OK] saved to: results/...

---

4.Current Status

✔ MVP runnable
✔ Unified algorithm interface
✔ Structured experiment output
✔ Git-based collaborative development

Next steps:
Integrate real data management
Support multiple algorithm plugins
Add experiment comparison tools

---

5.Agent API Integration (new)

This repository now includes an Agent-ready API layer that wraps existing pipeline functions.

Main files:
- src/agent/api.py
- src/agent/service.py
- src/agent/skills.py
- src/agent/openai_agent.py
- run_agent_api.py
- skills.json

Key endpoints:
- POST /api/set-algorithm
- POST /api/set-mode
- POST /api/set-preprocess
- POST /api/upload-data
- POST /api/set-step
- POST /api/run-pipeline
- GET /api/generate-chart
- GET /api/generate-report
- POST /api/chat
- POST /api/chat/reset

Highlights:
- Streamlit sidebar now has built-in Agent chat panel in app.py.
- Chart API now supports: 时域, 频域, 地形图.
- Chat supports session memory with session_id (supports follow-ups like "再跑一次，把滤波改成7-35Hz").

Quick start:
1) Install dependencies
pip install fastapi uvicorn openai matplotlib pydantic streamlit plotly

2) Configure model key (PowerShell)
$env:OPENAI_API_KEY="your_key"

DeepSeek (OpenAI-compatible) recommended setup:
$env:DEEPSEEK_API_KEY="your_deepseek_key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:AGENT_LLM_MODEL="deepseek-chat"

3) Start API service
python run_agent_api.py

4) Open docs
http://localhost:8510/docs

5) Agent-generated chart/report static files
http://localhost:8510/results

Example chat request:
POST /api/chat
{
	"user_input": "帮我用EEGNet，设置8-30Hz，跑完整流程并生成地形图和markdown报告",
	"session_id": "demo-user-001"
}

Follow-up request with memory:
POST /api/chat
{
	"user_input": "再跑一次，把滤波改成7-35Hz",
	"session_id": "demo-user-001"
}

