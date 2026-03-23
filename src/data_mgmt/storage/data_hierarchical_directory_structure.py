import os
import json
import uuid
import getpass
from datetime import datetime
from pathlib import Path

# =========================
# 路径定义
# =========================
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parents[2]

DATA_DIR = str(_PROJECT_ROOT / "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
FEATURE_DIR = os.path.join(DATA_DIR, "feature")

META_FILE = os.path.join(DATA_DIR, "meta.json")

# 日志系统
LOG_DIR = os.path.join(DATA_DIR, "logs")
EXP_INDEX_FILE = os.path.join(DATA_DIR, "experiments.json")

# =========================
# 初始化
# =========================

def init_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if not os.path.exists(META_FILE):
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

    if not os.path.exists(EXP_INDEX_FILE):
        with open(EXP_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4, ensure_ascii=False)


# =========================
# 基础数据接口（完全兼容）
# =========================

def generate_data_id():
    return str(uuid.uuid4())


def load_meta():
    with open(META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(meta):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


def save_raw(data, filename):
    init_dirs()

    data_id = generate_data_id()
    filepath = os.path.join(RAW_DIR, f"{data_id}_{filename}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(data)

    meta = load_meta()
    meta.append({
        "data_id": data_id,
        "type": "raw",
        "file": filepath,
        "time": datetime.now().isoformat()
    })
    save_meta(meta)

    return data_id


def save_feature(data_id, feature_data):
    init_dirs()

    filepath = os.path.join(FEATURE_DIR, f"{data_id}_feature.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(feature_data, f, indent=4)

    meta = load_meta()
    meta.append({
        "data_id": data_id,
        "type": "feature",
        "file": filepath,
        "time": datetime.now().isoformat()
    })
    save_meta(meta)

    return filepath


# =========================
# 🧠 实验日志系统（核心升级）
# =========================

def generate_exp_id():
    return f"exp_{uuid.uuid4().hex[:12]}"


def log_experiment(
    exp_name: str,
    data_id: str = None,
    parameters: dict = None,
    results: dict = None,
    status: str = "success",
    notes: str = "",
    start_time: str = None,
    end_time: str = None
):
    init_dirs()

    exp_id = generate_exp_id()
    user = getpass.getuser()

    now = datetime.now().isoformat()
    start_time = start_time or now
    end_time = end_time or now

    exp_log = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "user": user,
        "data_id": data_id,
        "start_time": start_time,
        "end_time": end_time,
        "parameters": parameters or {},
        "results": results or {},
        "status": status,
        "notes": notes
    }

    log_file = os.path.join(LOG_DIR, f"{exp_id}.json")

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(exp_log, f, indent=4, ensure_ascii=False)

    # 更新索引
    exp_list = []
    if os.path.exists(EXP_INDEX_FILE):
        with open(EXP_INDEX_FILE, "r", encoding="utf-8") as f:
            exp_list = json.load(f)

    exp_list.append({
        "exp_id": exp_id,
        "exp_name": exp_name,
        "user": user,
        "data_id": data_id,
        "time": start_time,
        "status": status,
        "log_file": log_file
    })

    with open(EXP_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(exp_list, f, indent=4, ensure_ascii=False)

    return exp_id


def get_exp_log(exp_id: str):
    path = os.path.join(LOG_DIR, f"{exp_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_experiments():
    if not os.path.exists(EXP_INDEX_FILE):
        return []
    with open(EXP_INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)