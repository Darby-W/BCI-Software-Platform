#实现数据读取与查询接口
import numpy as np
import pandas as pd
import mne
import os
import json
from datetime import datetime
from typing import Tuple, Dict, List
from pathlib import Path

# ===================== 基础配置 =====================
# 动态获取默认数据目录（无需手动配置）
_CURRENT_DIR = Path(__file__).resolve().parent  # .../src/data_mgmt/query/
_PROJECT_ROOT = _CURRENT_DIR.parents[2]  # 项目根目录
DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "src" / "data_mgmt" / "data_tools" / "third_party_device_data")

SUPPORT_FORMATS = ["EDF", "CSV"]
DEFAULT_SAMPLE_RATE = 250
EVENT_MAP = {"left_hand": 1, "right_hand": 2, "rest": 0, "eye_blink": 3}
LABEL_COLUMNS = ["label", "task", "event", "annotation", "target"]

# ===================== 工具函数 =====================
def init_data_dir(data_dir: str):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for fmt in SUPPORT_FORMATS:
        fmt_dir = os.path.join(data_dir, fmt.lower())
        if not os.path.exists(fmt_dir):
            os.makedirs(fmt_dir)

def validate_path(path: str) -> bool:
    try:
        test_file = os.path.join(path, f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except:
        return False

# ===================== 核心数据管理类（data_mgmt） =====================
class BCIDataSystem:
    def __init__(self, data_dir: str = None):
        # 支持初始化时传入目录，便于pipeline调用
        if data_dir is None:
            print("请配置第三方设备数据存储目录（直接回车使用默认目录）")
            user_data_dir = input(f"默认目录：{DEFAULT_DATA_DIR} → 输入自定义目录：").strip()
            self.data_dir = user_data_dir if user_data_dir else DEFAULT_DATA_DIR
        else:
            self.data_dir = data_dir

       # if not validate_path(self.data_dir):
       #    raise PermissionError(f"目录 {self.data_dir} 无读写权限！请更换目录")
        init_data_dir(self.data_dir)
        self.data_map = self._auto_scan_data()
        if not self.data_map:
            print(f"目录 {self.data_dir} 下未找到EDF/CSV格式的第三方设备数据")
        self.target_format = None

    def _auto_scan_data(self) -> Dict:
        data_map = {}
        idx = 1
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith((".edf", ".csv")):
                    data_id = f"exp_{str(idx).zfill(3)}"
                    fmt = "EDF" if file.lower().endswith(".edf") else "CSV"
                    file_path = os.path.abspath(os.path.join(root, file))
                    data_map[data_id] = {
                        "format": fmt,
                        "path": file_path,
                        "file_name": file
                    }
                    idx += 1
        return data_map

    # ===================== 任务2要求的接口 =====================
    def query_data(self) -> List[str]:
        """
        查询所有可用的数据 ID 列表
        返回:
            list: 所有可用的 data_id 列表
        """
        return list(self.data_map.keys())

    def load_feature(self, data_id: str, tmin: float = 0, tmax: float = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        根据 data_id 加载数据，返回标准结构 X, y, meta
        参数:
            data_id: 数据唯一标识
            tmin/tmax: 可选，裁剪时间段（秒）
        返回:
            X: 特征矩阵 (时间点 × 通道数)
            y: 标签数组 (时间点,)
            meta: 元数据字典
        """
        if data_id not in self.data_map:
            raise ValueError(f"数据ID {data_id} 不存在！可用ID：{list(self.data_map.keys())}")

        data_info = self.data_map[data_id]
        file_path = data_info["path"]
        raw_format = data_info["format"]

        if raw_format == "EDF":
            data = self._read_edf(file_path, data_id, tmin, tmax)
        elif raw_format == "CSV":
            data = self._read_csv(file_path, data_id)
        else:
            raise ValueError(f"不支持的原始格式：{raw_format}")

        return data["X"], data["y"], data["meta"]

    # ===================== 内部读取方法 =====================
    def _read_edf(self, file_path: str, data_id: str, tmin: float, tmax: float) -> Dict:
        try:
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            if tmax is not None and tmax > tmin:
                raw.crop(tmin=tmin, tmax=tmax)
            raw.load_data()
            X = raw.get_data().T
            y = np.zeros(X.shape[0], dtype=int)
            if raw.annotations:
                sfreq = raw.info["sfreq"]
                for ann in raw.annotations:
                    start_idx = int(ann["onset"] * sfreq)
                    end_idx = int((ann["onset"] + ann["duration"]) * sfreq)
                    y[start_idx:end_idx] = EVENT_MAP.get(ann["description"], 0)
            meta = {
                "data_id": data_id,
                "file_name": self.data_map[data_id]["file_name"],
                "channels": raw.ch_names,
                "sampling_rate": raw.info["sfreq"],
                "total_time": raw.n_times / raw.info["sfreq"],
                "raw_format": "EDF",
                "file_path": file_path
            }
            return {"X": X, "y": y, "meta": meta}
        except Exception as e:
            raise ValueError(f"读取EDF文件失败：{str(e)}")

    def _read_csv(self, file_path: str, data_id: str) -> Dict:
        try:
            df = pd.read_csv(file_path)
            channel_cols = [col for col in df.columns if any(key in col.lower() for key in ["eeg", "ch","f"])]
            if not channel_cols:
                raise ValueError("未找到EEG通道列（列名建议包含EEG/Ch）")
            # ===== 判断是否是 trial 展平格式 =====
            if any("_t" in col for col in channel_cols):
                print("检测到 trial 展平格式CSV，自动还原3D结构")

                num_trials = df.shape[0]
                num_channels = 22
                num_timepoints = 1000

                X_flat = df[channel_cols].values

                X = X_flat.reshape(num_trials, num_channels, num_timepoints)

            else:
                # 原始时间序列格式（备用）
                X = df[channel_cols].values
            y = np.zeros(X.shape[0], dtype=int)
            label_col = next((col for col in LABEL_COLUMNS if col in df.columns), None)
            if label_col:
                y = df[label_col].values.astype(int)
            meta = {
                "data_id": data_id,
                "file_name": self.data_map[data_id]["file_name"],
                "channels": channel_cols,
                "sampling_rate": DEFAULT_SAMPLE_RATE,
                "total_time": len(df) / DEFAULT_SAMPLE_RATE,
                "raw_format": "CSV",
                "file_path": file_path
            }
            return {"X": X, "y": y, "meta": meta}
        except Exception as e:
            raise ValueError(f"读取CSV文件失败：{str(e)}")

    # ===================== 导出与预览功能（保留原有能力） =====================
    def preview_data(self, data_id: str):
        try:
            data = self.load_feature(data_id, tmin=0, tmax=10)
            X, y, meta = data
            print(f"\n===== 数据 {data_id} 预览 =====")
            print(f"原始文件：{meta['file_name']}")
            print(f"数据形状（时间点×通道数）：{X.shape}")
            print(f"采样率：{meta['sampling_rate']}Hz")
            print(f"数据时长（预览）：{meta['total_time']:.2f}秒")
            print(f"数据范围：{X.min():.4f} ~ {X.max():.4f}")
            print(f"非零标签数量：{np.count_nonzero(y)}")
        except Exception as e:
            print(f"预览失败：{str(e)}")

    def export_data(self, data_id: str, save_path: str = None, tmin: float = 0, tmax: float = None) -> str:
        try:
            if not self.target_format:
                raise ValueError("请先输入指令指定输出格式！")
            X, y, meta = self.load_feature(data_id, tmin, tmax)
            if save_path is None:
                save_dir = os.path.join(self.data_dir, self.target_format.lower())
                save_name = f"{data_id}_export_{datetime.now().strftime('%Y%m%d%H%M%S')}.{self.target_format.lower()}"
                save_path = os.path.join(save_dir, save_name)
            else:
                if not save_path.lower().endswith(self.target_format.lower()):
                    save_path += f".{self.target_format.lower()}"
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            if self.target_format == "EDF":
                self._export_edf(X, meta, save_path)
            elif self.target_format == "CSV":
                self._export_csv(X, y, meta, save_path)
            print(f"数据已导出至：{os.path.abspath(save_path)}")
            return save_path
        except PermissionError:
            raise ValueError(f"保存路径无写入权限！请更换路径（如桌面）")
        except FileNotFoundError:
            raise ValueError(f"原始数据文件不存在！路径：{meta.get('file_path', '未知')}")
        except Exception as e:
            raise ValueError(f"导出失败：{str(e)}")

    def _export_edf(self, X: np.ndarray, meta: Dict, save_path: str):
        data = X.T
        info = mne.create_info(
            ch_names=meta["channels"],
            sfreq=meta["sampling_rate"],
            ch_types=["eeg"] * len(meta["channels"])
        )
        raw = mne.io.RawArray(data, info)
        raw.export(save_path, fmt="edf", overwrite=True)

    def _export_csv(self, X: np.ndarray, y: np.ndarray, meta: Dict, save_path: str):
        df = pd.DataFrame(X, columns=meta["channels"])
        df["label"] = y
        df["sampling_rate"] = meta["sampling_rate"]
        df["data_id"] = meta["data_id"]
        df["raw_file"] = meta["file_name"]
        df.to_csv(save_path, index=False, encoding="utf-8")

    def parse_command(self, user_input: str) -> bool:
        user_input = user_input.strip().upper()
        for fmt in SUPPORT_FORMATS:
            if fmt in user_input:
                self.target_format = fmt
                print(f"已确认输出格式：{fmt}")
                return True
        print(f"无效指令！仅支持格式：{', '.join(SUPPORT_FORMATS)}")
        return False

    def batch_export(self, save_dir: str = None):
        if not self.target_format:
            raise ValueError("请先输入指令指定输出格式！")
        save_dir = save_dir or os.path.join(self.data_dir, f"batch_export_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        success_count = 0
        fail_list = []
        print(f"\n开始批量导出（目标格式：{self.target_format}），保存目录：{save_dir}")
        for data_id in self.data_map.keys():
            try:
                self.export_data(data_id, os.path.join(save_dir, f"{data_id}.{self.target_format.lower()}"))
                success_count += 1
            except Exception as e:
                fail_list.append(f"{data_id}：{str(e)}")
                print(f"{data_id} 导出失败：{str(e)}")
        print(f"\n批量导出完成：成功{success_count}个，失败{len(fail_list)}个")
        if fail_list:
            print("失败列表：")
            for fail in fail_list:
                print(f"  - {fail}")

# ===================== 交互式运行入口 =====================
def main():
    try:
        bci_system = BCIDataSystem()
        print("\n===== 脑机接口（BCI）数据管理系统 ===== v2.0")
        print(f"可用数据ID：{bci_system.query_data() or '无'}")
        while True:
            print("\n---------- 操作菜单 ----------")
            print("1. 指定输出格式（必填）")
            print("2. 预览数据（验证数据是否正确）")
            print("3. 导出单个数据")
            print("4. 批量导出所有数据")
            print("5. 退出系统")
            choice = input("\n请选择操作（1-5）：").strip()
            if choice == "1":
                cmd = input("请输入输出格式指令（如'EDF'、'CSV'）：")
                bci_system.parse_command(cmd)
            elif choice == "2":
                if not bci_system.query_data():
                    print("无可用数据！")
                    continue
                data_id = input(f"请输入要预览的数据ID（可用：{bci_system.query_data()}）：")
                bci_system.preview_data(data_id)
            elif choice == "3":
                if not bci_system.query_data():
                    print("无可用数据！")
                    continue
                data_id = input(f"请输入要导出的数据ID（可用：{bci_system.query_data()}）：")
                custom_path = input("请输入自定义保存路径（直接回车用默认路径）：").strip()
                try:
                    bci_system.export_data(data_id, custom_path if custom_path else None)
                except Exception as e:
                    print(e)
            elif choice == "4":
                custom_dir = input("请输入批量导出目录（直接回车用默认目录）：").strip()
                try:
                    bci_system.batch_export(custom_dir if custom_dir else None)
                except Exception as e:
                    print(e)
            elif choice == "5":
                print("👋 退出系统成功！")
                break
            else:
                print("无效选择，请输入1-5！")
    except Exception as e:
        print(f"系统初始化失败：{str(e)}")
        input("按回车键退出...")

if __name__ == "__main__":
    main()