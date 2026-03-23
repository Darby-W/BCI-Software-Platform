import os
import json
import uuid
import getpass
from datetime import datetime
from pathlib import Path

# 实现数据分层目录结构
# =========================
# 定义数据目录（动态计算项目根目录）
# =========================

_CURRENT_DIR = Path(__file__).resolve().parent  # .../src/data_mgmt/storage/
_PROJECT_ROOT = _CURRENT_DIR.parents[2]  # 项目根目录
DATA_DIR = str(_PROJECT_ROOT / "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
FEATURE_DIR = os.path.join(DATA_DIR, "feature")
META_FILE = os.path.join(DATA_DIR, "meta.json")

# =========================
# 实验日志系统目录（图片要求新增）
# =========================
LOG_DIR = os.path.join(DATA_DIR, "logs")
EXP_INDEX_FILE = os.path.join(DATA_DIR, "experiments.json")

# =========================
# 初始化目录结构
# =========================

def init_dirs():
    # 创建 data/raw 文件夹
    os.makedirs(RAW_DIR, exist_ok=True)

    # 创建 data/feature 文件夹
    os.makedirs(FEATURE_DIR, exist_ok=True)

    # 如果 meta.json 不存在就创建
    if not os.path.exists(META_FILE):
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

# =========================
# 初始化日志目录（图片要求）
# =========================
def init_log_system():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(EXP_INDEX_FILE):
        with open(EXP_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)

# =========================
# 生成唯一 data_id
# =========================

def generate_data_id():
    return str(uuid.uuid4())

# =========================
# 生成实验ID（图片要求）
# =========================
def generate_exp_id():
    return f"exp_{uuid.uuid4().hex[:12]}"

# =========================
# 读取 meta.json
# =========================

def load_meta():
    with open(META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# 保存 meta.json
# =========================

def save_meta(meta):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

# =========================
# 保存 raw 数据
# =========================

def save_raw(data, filename):

    # 初始化目录
    init_dirs()

    # 生成 data_id
    data_id = generate_data_id()

    # 构造 raw 文件路径
    filepath = os.path.join(RAW_DIR, f"{data_id}_{filename}")

    # 保存数据
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(data)

    # 读取 meta.json
    meta = load_meta()

    # 添加记录
    meta.append({
        "data_id": data_id,
        "type": "raw",
        "file": filepath,
        "time": datetime.now().isoformat()
    })

    # 保存 meta.json
    save_meta(meta)

    return data_id

# =========================
# 保存 feature 数据
# =========================

def save_feature(data_id, feature_data):

    # 初始化目录
    init_dirs()

    # 构造 feature 文件路径
    filepath = os.path.join(FEATURE_DIR, f"{data_id}_feature.json")

    # 保存 feature
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(feature_data, f, indent=4)

    # 读取 meta.json
    meta = load_meta()

    # 添加记录
    meta.append({
        "data_id": data_id,
        "type": "feature",
        "file": filepath,
        "time": datetime.now().isoformat()
    })

    # 保存 meta.json
    save_meta(meta)

    return filepath

# =========================
# =========================
# 实验日志系统（严格按照图片要求实现）
# =========================
# =========================

def log_experiment(
    exp_name: str,
    data_id: str = None,
    parameters: dict = None,
    results: dict = None,
    status: str = "success",
    notes: str = ""
):
    """
    按图片要求：完整实验日志记录
    包含：实验ID、实验名称、用户名、时间、参数、结果、状态、关联数据ID、备注
    """
    init_dirs()
    init_log_system()

    exp_id = generate_exp_id()
    now = datetime.now().isoformat()
    user = getpass.getuser()

    # 单条实验完整日志
    exp_log = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "user": user,
        "data_id": data_id,
        "start_time": now,
        "parameters": parameters or {},
        "results": results or {},
        "status": status,
        "notes": notes
    }

    # 保存到独立日志文件
    log_file = os.path.join(LOG_DIR, f"{exp_id}.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(exp_log, f, ensure_ascii=False, indent=4)

    # 更新实验索引
    exp_list = []
    if os.path.exists(EXP_INDEX_FILE):
        with open(EXP_INDEX_FILE, "r", encoding="utf-8") as f:
            exp_list = json.load(f)

    exp_list.append({
        "exp_id": exp_id,
        "exp_name": exp_name,
        "user": user,
        "data_id": data_id,
        "time": now,
        "status": status,
        "log_file": log_file
    })

    with open(EXP_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(exp_list, f, ensure_ascii=False, indent=4)

    return exp_id

def get_exp_log(exp_id: str):
    """根据实验ID获取完整日志"""
    log_path = os.path.join(LOG_DIR, f"{exp_id}.json")
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_all_experiments():
    """获取所有实验记录"""
    if not os.path.exists(EXP_INDEX_FILE):
        return []
    with open(EXP_INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)