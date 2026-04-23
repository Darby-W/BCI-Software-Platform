import os

from src.data_mgmt.query import BCIDataSystem
from src.utils.paths import default_third_party_data_dir, resolve_project_path

configured_data_dir = os.getenv("BCI_DATA_DIR")
resolved_data_dir = resolve_project_path(configured_data_dir) if configured_data_dir else default_third_party_data_dir()

bci = BCIDataSystem(data_dir=str(resolved_data_dir))

print("可用数据：")
print(bci.query_data())

X, y, meta = bci.load_feature("exp_001")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("meta:", meta)