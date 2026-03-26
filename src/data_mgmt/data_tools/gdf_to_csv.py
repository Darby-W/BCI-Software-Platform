"""
gdf_to_csv.py - 支持训练集和测试集转换
训练集（T文件）：提取事件标签，按trial切分
测试集（E文件）：导出连续EEG数据
"""

import os
import sys
from pathlib import Path
import mne
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===== 自动路径配置 =====
# 获取当前文件所在目录的绝对路径
CURRENT_FILE_DIR = Path(__file__).resolve().parent
# 定位到项目根目录（向上多个层级）
PROJECT_ROOT = CURRENT_FILE_DIR.parents[2]  # 从 ...data_mgmt/data_tools -> 项目根
# 输出目录：src/data_mgmt/data_tools/third_party_device_data
CSV_OUTPUT_DIR = CURRENT_FILE_DIR / "third_party_device_data"

# 输入目录：优先使用环境变量，否则使用项目根目录下的 datasets 文件夹
GDF_INPUT_DIR_ENV = os.getenv('GDF_INPUT_DIR')
if GDF_INPUT_DIR_ENV:
    GDF_INPUT_DIR = Path(GDF_INPUT_DIR_ENV)
else:
    # 默认位置：项目根目录/datasets（递归处理其下所有 .gdf 文件）
    GDF_INPUT_DIR = PROJECT_ROOT / "datasets"

# 创建输出目录
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BCI2A_CHANNEL_MAPPING = {
    'EEG-Fz': 'Fz',
    'EEG-0': 'FC3',
    'EEG-1': 'FC1',
    'EEG-2': 'FCz',
    'EEG-3': 'FC2',
    'EEG-4': 'FC4',
    'EEG-5': 'C5',
    'EEG-C3': 'C3',
    'EEG-6': 'C1',
    'EEG-Cz': 'Cz',
    'EEG-7': 'C2',
    'EEG-C4': 'C4',
    'EEG-8': 'C6',
    'EEG-9': 'CP3',
    'EEG-10': 'CP1',
    'EEG-11': 'CPz',
    'EEG-12': 'CP2',
    'EEG-13': 'CP4',
    'EEG-14': 'P7',
    'EEG-Pz': 'Pz',
    'EEG-15': 'P5',
    'EEG-16': 'P3',
    'EEG-17': 'P1'
}

SAMPLING_RATE = 250

# 运动想象事件ID映射
EVENT_ID_MAP = {
    769: 0,  # 左手
    770: 1,  # 右手
    771: 2,  # 脚
    772: 3,  # 舌头
}


def is_training_file(filename: str) -> bool:
    """
    判断是否为训练集文件（T结尾）
    BCI Competition IV 2a 命名规范: A01T.gdf 是训练集, A01E.gdf 是测试集
    
    Args:
        filename: 文件名
    
    Returns:
        True: 训练集，False: 测试集
    """
    name_without_ext = Path(filename).stem
    # 检查文件名是否以 T 结尾（如 A01T, A02T 等）
    return name_without_ext.endswith('T')


def convert_training_file(gdf_file_path: str, csv_save_path: str,
                          t_start: float = 0.5, t_end: float = 4.5):
    """
    转换训练集文件（带事件标签，按trial切分）
    
    Args:
        gdf_file_path: GDF文件路径
        csv_save_path: CSV保存路径
        t_start: trial开始时间（秒），默认刺激后0.5秒
        t_end: trial结束时间（秒），默认刺激后4.5秒
    """
    raw = mne.io.read_raw_gdf(gdf_file_path, preload=True, verbose=False)

    # ===== 通道选择 =====
    available_channels = raw.ch_names
    channel_mapping = {
        raw_ch: BCI2A_CHANNEL_MAPPING[raw_ch]
        for raw_ch in available_channels
        if raw_ch in BCI2A_CHANNEL_MAPPING
    }

    if not channel_mapping:
        print(f" ⚠️ {os.path.basename(gdf_file_path)} 无匹配通道，跳过")
        return

    raw.pick(list(channel_mapping.keys()))
    eeg_data = raw.get_data()

    # ===== 提取事件 =====
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    print(f"\n📄 {os.path.basename(gdf_file_path)} [训练集]")
    print(f"   事件字典: {event_dict}")

    # 找到运动想象事件 (769-772) 在MNE里的真实编码
    target_event_codes = []
    for k, v in event_dict.items():
        if k in ['769', '770', '771', '772']:
            target_event_codes.append(v)

    if len(target_event_codes) == 0:
        print(f"   ⚠️ 未找到运动想象事件 (769-772)，跳过")
        return

    valid_events = events[np.isin(events[:, 2], target_event_codes)]

    if len(valid_events) == 0:
        print(f"   ⚠️ 无有效事件")
        return

    # 反查：MNE编码 → 原始事件ID
    reverse_event_dict = {v: k for k, v in event_dict.items()}

    # ===== trial 切分参数 =====
    fs = SAMPLING_RATE
    start_offset = int(t_start * fs)
    end_offset = int(t_end * fs)

    X = []
    y = []
    event_names = []

    # ===== 提取trial =====
    for event_time, _, event_code in valid_events:
        # 转回真实事件ID
        real_event = int(reverse_event_dict[event_code])
        label = EVENT_ID_MAP.get(real_event, -1)

        if label == -1:
            continue

        start_idx = int(event_time + start_offset)
        end_idx = int(event_time + end_offset)

        if end_idx <= eeg_data.shape[1] and start_idx >= 0:
            epoch = eeg_data[:, start_idx:end_idx]
            X.append(epoch)
            y.append(label)
            event_names.append(str(real_event))

    if len(X) == 0:
        print(f"   ⚠️ 无有效trial")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"   ✅ 成功提取 {len(X)} 个trial")
    print(f"   数据shape: {X.shape}")
    print(f"   标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ===== 保存CSV =====
    rows = []
    for i in range(len(X)):
        row = {
            "trial_id": i,
            "label": int(y[i]),
            "event_id": event_names[i],
            "dataset_type": "training"
        }
        for ch in range(X.shape[1]):
            for t in range(X.shape[2]):
                row[f"ch{ch}_t{t}"] = X[i, ch, t]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_save_path, index=False)
    print(f"   💾 保存成功: {csv_save_path}")


def convert_evaluation_file(gdf_file_path: str, csv_save_path: str):
    """
    转换测试集文件（连续EEG数据，不提取trial）
    
    Args:
        gdf_file_path: GDF文件路径
        csv_save_path: CSV保存路径
    """
    raw = mne.io.read_raw_gdf(gdf_file_path, preload=True, verbose=False)

    # ===== 通道选择 =====
    available_channels = raw.ch_names
    channel_mapping = {
        raw_ch: BCI2A_CHANNEL_MAPPING[raw_ch]
        for raw_ch in available_channels
        if raw_ch in BCI2A_CHANNEL_MAPPING
    }

    if not channel_mapping:
        print(f" ⚠️ {os.path.basename(gdf_file_path)} 无匹配通道，跳过")
        return

    raw.pick(list(channel_mapping.keys()))
    eeg_data = raw.get_data()

    print(f"\n📄 {os.path.basename(gdf_file_path)} [测试集]")
    print(f"   数据形状: {eeg_data.shape}")
    print(f"   时长: {eeg_data.shape[1] / SAMPLING_RATE:.1f} 秒")

    # 尝试提取事件信息（如果有的话）
    try:
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        if len(events) > 0:
            print(f"   事件字典: {event_dict}")
            print(f"   事件数量: {len(events)}")
    except:
        print(f"   无事件信息")

    # ===== 导出连续数据 =====
    rows = []
    
    # 按时间点导出
    for t in range(eeg_data.shape[1]):
        row = {
            "time": t / SAMPLING_RATE,  # 时间（秒）
            "sample_index": t,
            "dataset_type": "evaluation"
        }
        for ch in range(eeg_data.shape[0]):
            # 使用通道名称作为列名
            ch_name = f"ch{ch}"
            row[ch_name] = eeg_data[ch, t]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_save_path, index=False)
    print(f"   ✅ 导出 {len(rows)} 个时间点")
    print(f"   💾 保存成功: {csv_save_path}")


def convert_gdf_to_csv(gdf_file_path: str, csv_save_path: str,
                       t_start: float = 0.5, t_end: float = 4.5):
    """
    根据文件名自动判断是训练集还是测试集
    
    Args:
        gdf_file_path: GDF文件路径
        csv_save_path: CSV保存路径
        t_start: 训练集trial开始时间（秒）
        t_end: 训练集trial结束时间（秒）
    """
    filename = os.path.basename(gdf_file_path)
    
    if is_training_file(filename):
        convert_training_file(gdf_file_path, csv_save_path, t_start, t_end)
    else:
        convert_evaluation_file(gdf_file_path, csv_save_path)


def batch_convert_all_gdf(t_start: float = 0.5, t_end: float = 4.5):
    """
    批量转换所有GDF文件
    
    Args:
        t_start: 训练集trial开始时间（秒）
        t_end: 训练集trial结束时间（秒）
    """
    gdf_input_dir_str = str(GDF_INPUT_DIR)
    csv_output_dir_str = str(CSV_OUTPUT_DIR)
    
    if not os.path.exists(gdf_input_dir_str):
        print(f"❌ GDF输入目录不存在: {gdf_input_dir_str}")
        print(f"   请设置环境变量 GDF_INPUT_DIR 或在 {PROJECT_ROOT}/datasets 放置数据文件")
        return

    os.makedirs(csv_output_dir_str, exist_ok=True)

    gdf_files = sorted(GDF_INPUT_DIR.rglob('*.gdf'))
    if not gdf_files:
        print(f"❌ 在 {GDF_INPUT_DIR} 下未找到任何 .gdf 文件")
        return

    print(f"\n📁 找到 {len(gdf_files)} 个GDF文件")
    print(f"📁 输入目录: {GDF_INPUT_DIR}")
    print(f"📁 输出目录: {CSV_OUTPUT_DIR}")
    print(f"📌 训练集trial时间窗口: {t_start}s - {t_end}s")
    print("-" * 60)

    training_count = 0
    evaluation_count = 0
    fail_count = 0

    for gdf_path in gdf_files:
        relative_path = gdf_path.relative_to(GDF_INPUT_DIR)
        csv_relative_path = relative_path.with_suffix('.csv')
        csv_path = CSV_OUTPUT_DIR / csv_relative_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            convert_gdf_to_csv(str(gdf_path), str(csv_path), t_start, t_end)
            if is_training_file(gdf_path.name):
                training_count += 1
            else:
                evaluation_count += 1
        except Exception as e:
            fail_count += 1
            print(f"   ❌ 转换失败: {e}")
    
    print("-" * 60)
    print(f"✅ 批量转换完成!")
    print(f"   - 训练集: {training_count} 个文件")
    print(f"   - 测试集: {evaluation_count} 个文件")
    if fail_count > 0:
        print(f"   - 失败: {fail_count} 个文件")


if __name__ == "__main__":
    # 可以修改trial时间窗口参数
    # t_start=0.5, t_end=4.5 是标准设置
    # 如果想调整，可以修改下面的参数
    batch_convert_all_gdf(t_start=0.5, t_end=4.5)