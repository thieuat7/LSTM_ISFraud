import pandas as pd
import numpy as np
import os

# === Cấu hình ===
FOLDER_NAME = os.path.join("data_clear2")  # thư mục chứa file CSV
FULL_TRAIN_FILE = os.path.join(FOLDER_NAME, "train_cleaned.csv")

TIME_COL = "TransactionDT"
TARGET_COL = "isFraud"

# === 1. Đọc dữ liệu Train gốc ===
print("⏳ Đang đọc dữ liệu Train gốc...")
df_full = pd.read_csv(FULL_TRAIN_FILE)

# === 2. Sắp xếp theo thời gian ===
df_full = df_full.sort_values(by=TIME_COL).reset_index(drop=True)

# === 3. Chia 80% - 20% ===
split_idx = int(len(df_full) * 0.8)
df_train_new = df_full.iloc[:split_idx].copy()
df_val = df_full.iloc[split_idx:].copy()

print("✅ Đã chia dữ liệu:")
print(f"   - Train mới (80%): {len(df_train_new)} mẫu")
print(f"   - Validation (20%): {len(df_val)} mẫu")

# === 4. Lưu file kết quả ===
df_train_new.to_csv(os.path.join(FOLDER_NAME, "train_split.csv"), index=False)
df_val.to_csv(os.path.join(FOLDER_NAME, "val_split.csv"), index=False)

print("\n💾 Đã lưu 'train_split.csv' và 'val_split.csv'.")
print(f"   -> Validation có {df_val[TARGET_COL].sum()} giao dịch gian lận để test.")
