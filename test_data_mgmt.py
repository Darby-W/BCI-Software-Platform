import os
from pathlib import Path

from src.data_mgmt.query import BCIDataSystem

project_root = Path(__file__).resolve().parent
configured_data_dir = os.getenv("BCI_DATA_DIR") or "src/data_mgmt/data_tools/third_party_device_data"
resolved_data_dir = Path(configured_data_dir)
if not resolved_data_dir.is_absolute():
	resolved_data_dir = project_root / resolved_data_dir

bci = BCIDataSystem(data_dir=str(resolved_data_dir))

print("可用数据：")
print(bci.query_data())

X, y, meta = bci.load_feature("exp_001")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("meta:", meta)