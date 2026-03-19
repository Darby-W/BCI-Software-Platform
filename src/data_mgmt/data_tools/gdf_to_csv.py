import os
import mne
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

GDF_INPUT_DIR = r"D:\public_dataset_BCI\BCICIV_2a_gdf"
CSV_OUTPUT_DIR = r"third_party_device_data"

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

# 🔥 只保留真实分类标签
EVENT_ID_MAP = {
    769: 0,
    770: 1,
    771: 2,
    772: 3,
}


def convert_gdf_to_csv(gdf_file_path: str, csv_save_path: str):
    raw = mne.io.read_raw_gdf(gdf_file_path, preload=True, verbose=False)

    # ===== 通道选择 =====
    available_channels = raw.ch_names
    channel_mapping = {
        raw_ch: BCI2A_CHANNEL_MAPPING[raw_ch]
        for raw_ch in available_channels
        if raw_ch in BCI2A_CHANNEL_MAPPING
    }

    if not channel_mapping:
        print(f"❌ {gdf_file_path} 无匹配通道")
        return

    raw.pick(list(channel_mapping.keys()))
    eeg_data = raw.get_data()

    # ===== 🔥 正确提取事件 =====
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    print(f"\n📄 {os.path.basename(gdf_file_path)} 事件字典：{event_dict}")

    # 🔥 找到 769–772 在MNE里的真实编码
    target_event_codes = []
    for k, v in event_dict.items():
        if k in ['769', '770', '771', '772']:
            target_event_codes.append(v)

    if len(target_event_codes) == 0:
        print(f"❌ 没找到769–772事件")
        return

    valid_events = events[np.isin(events[:, 2], target_event_codes)]

    if len(valid_events) == 0:
        print(f"⚠️ 无有效事件")
        return

    # 🔥 反查：MNE编码 → 原始769
    reverse_event_dict = {v: k for k, v in event_dict.items()}

    # ===== trial 切分参数 =====
    fs = SAMPLING_RATE
    t_start = 0      # 可以改成 0.5 或 2
    t_end   = 4

    start_offset = int(t_start * fs)
    end_offset   = int(t_end * fs)

    X = []
    y = []

    # ===== 正确 trial 切分 =====
    for event_time, _, event_code in valid_events:

        # 🔥 转回真实事件ID（769）
        real_event = int(reverse_event_dict[event_code])
        label = EVENT_ID_MAP[real_event]

        start_idx = int(event_time + start_offset)
        end_idx   = int(event_time + end_offset)

        if end_idx <= eeg_data.shape[1]:
            epoch = eeg_data[:, start_idx:end_idx]

            X.append(epoch)
            y.append(label)

    if len(X) == 0:
        print(f"⚠️ 无有效trial")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"✅ 数据shape: {X.shape}, 标签分布: {np.bincount(y)}")

    # ===== 保存CSV（保持你原结构）=====
    rows = []
    for i in range(len(X)):
        row = {
            "trial_id": i,
            "label": int(y[i])
        }
        for ch in range(X.shape[1]):
            for t in range(X.shape[2]):
                row[f"ch{ch}_t{t}"] = X[i, ch, t]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_save_path, index=False)

    print(f"✅ 保存成功: {csv_save_path}")


def batch_convert_all_gdf():
    if not os.path.exists(GDF_INPUT_DIR):
        print(f"GDF输入目录 {GDF_INPUT_DIR}不存在")
        return

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    gdf_files = [f for f in os.listdir(GDF_INPUT_DIR) if f.lower().endswith('.gdf')]
    if not gdf_files:
        print("未找到任何.gdf文件")
        return

    for gdf_file in gdf_files:
        gdf_path = os.path.join(GDF_INPUT_DIR, gdf_file)
        csv_filename = os.path.splitext(gdf_file)[0] + ".csv"
        csv_path = os.path.join(CSV_OUTPUT_DIR, csv_filename)

        convert_gdf_to_csv(gdf_path, csv_path)


if __name__ == "__main__":
    batch_convert_all_gdf()