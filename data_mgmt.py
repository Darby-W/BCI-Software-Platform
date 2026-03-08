import os
import json
import uuid
from datetime import datetime

# =========================
# 定义数据目录
# =========================

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
FEATURE_DIR = os.path.join(DATA_DIR, "feature")
META_FILE = os.path.join(DATA_DIR, "meta.json")

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
# 生成唯一 data_id
# =========================

def generate_data_id():
    return str(uuid.uuid4())


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