import pandas as pd
import numpy as np
import gc
import os
from sklearn.preprocessing import LabelEncoder

# =========================================
# 1. CẤU HÌNH & DANH SÁCH ĐẶC TRƯNG
# =========================================
PATH_TRANSACTION = "data1/train_transaction.csv"
PATH_IDENTITY = "data1/train_identity.csv"
OUTPUT_DIR = "data_clear"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_filtered_processed.csv")

# Danh sách các đặc trưng QUAN TRỌNG cần giữ lại (từ bạn cung cấp)
# Lưu ý: Tôi đã sửa lỗi chính tả nhỏ ở 'P_emaildomain''R_emaildomain' thành hai phần tử riêng biệt.
FEATURES_TO_KEEP = [
    'TransactionID', 'isFraud', 'TransactionDT', # Giữ lại các cột cốt lõi bắt buộc
    'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card5','card6',
    'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
    'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9','C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5','D10', 'D11', 'D15',
    'M1', 'M2', 'M3', 'M4', 'M6', 'M7', 'M8','M9',
    'V1', 'V3', 'V4', 'V6', 'V8', 'V11', 'V13', 'V14', 'V17','V20', 'V23', 'V26', 'V27',
    'V30', 'V36', 'V37', 'V40', 'V41','V44', 'V47', 'V48', 'V54', 'V56', 'V59', 'V62',
    'V65', 'V67','V68', 'V70', 'V76', 'V78', 'V80', 'V82', 'V86', 'V88', 'V89','V91',
    'V107', 'V108', 'V111', 'V115', 'V117', 'V120', 'V121','V123', 'V124', 'V127',
    'V129', 'V130', 'V136', 'V138', 'V139','V142', 'V147', 'V156', 'V160', 'V162',
    'V165', 'V166', 'V169','V171', 'V173', 'V175', 'V176', 'V178', 'V180', 'V182',
    'V185','V187', 'V188', 'V198', 'V203', 'V205', 'V207', 'V209', 'V210','V215',
    'V218', 'V220', 'V221', 'V223', 'V224', 'V226', 'V228','V229', 'V234', 'V235',
    'V238', 'V240', 'V250', 'V252', 'V253','V257', 'V258', 'V260', 'V261', 'V264',
    'V266', 'V267', 'V271','V274', 'V277', 'V281', 'V283', 'V284', 'V285', 'V286',
    'V289','V291', 'V294', 'V296', 'V297', 'V301', 'V303', 'V305', 'V307','V309',
    'V310', 'V314', 'V320',
    'id_01', 'id_02', 'id_03', 'id_04','id_05', 'id_06', 'id_09', 'id_10', 'id_11',
    'id_12', 'id_13','id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28',
    'id_29', 'id_31', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType','DeviceInfo'
]

# =========================================
# 2. HÀM TỐI ƯU BỘ NHỚ
# =========================================
def reduce_mem_usage(df, verbose=True):
    # ... (Giữ nguyên hàm này như trước) ...
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print('   📉 Giảm bộ nhớ: {:5.2f} Mb ({:.1f}% giảm)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# =========================================
# 3. QUY TRÌNH XỬ LÝ CHÍNH
# =========================================
print("⏳ [1/6] Đang đọc file Transaction...")
df_trans = pd.read_csv(PATH_TRANSACTION)

print("⏳ [2/6] Đang đọc file Identity...")
df_id = pd.read_csv(PATH_IDENTITY)

print("\n⏳ [3/6] Đang gộp (Merge) và lọc cột ngay lập tức...")
df_merged = pd.merge(df_trans, df_id, on='TransactionID', how='left')
del df_trans, df_id
gc.collect()

# --- LỌC CỘT QUAN TRỌNG ---
# Chỉ giữ lại các cột có trong danh sách FEATURES_TO_KEEP và thực sự tồn tại trong df_merged
final_cols = [col for col in FEATURES_TO_KEEP if col in df_merged.columns]
# Đảm bảo giữ lại 'isFraud' nếu nó tồn tại (cho tập train)
if 'isFraud' not in final_cols and 'isFraud' in df_merged.columns:
    final_cols.append('isFraud')

df_merged = df_merged[final_cols]
print(f"   ✅ Đã lọc dữ liệu. Shape hiện tại: {df_merged.shape}")
df_merged = reduce_mem_usage(df_merged)

# --- PHÂN LOẠI CỘT ĐỂ XỬ LÝ ---
# Xác định lại các cột cần xử lý dựa trên danh sách ĐÃ LỌC
# Lấy tất cả các cột kiểu object (chữ) còn lại để xử lý
cat_cols = df_merged.select_dtypes(include=['object']).columns.tolist()

# Phân loại thủ công dựa trên kinh nghiệm (bạn có thể điều chỉnh thêm)
# High cardinality: Nhiều giá trị khác nhau -> dùng Label Encoding
high_card_cols = [col for col in ['P_emaildomain', 'R_emaildomain', 'id_30', 'id_31', 'DeviceInfo'] if col in cat_cols]
# Low cardinality: Ít giá trị -> dùng One-Hot Encoding
low_card_cols = [col for col in cat_cols if col not in high_card_cols]

print(f"   ℹ️ Các cột sẽ Label Encoding: {high_card_cols}")
print(f"   ℹ️ Các cột sẽ One-Hot Encoding: {low_card_cols}")

# --- BƯỚC 4: Label Encoding ---
if len(high_card_cols) > 0:
    print(f"\n⏳ [4/6] Đang thực hiện Label Encoding...")
    for col in high_card_cols:
        df_merged[col] = df_merged[col].astype(str).fillna('unknown')
        le = LabelEncoder()
        df_merged[col] = le.fit_transform(df_merged[col])
        # Chuyển về kiểu dữ liệu nhỏ nhất có thể
        df_merged[col] = df_merged[col].astype('int32')

# --- BƯỚC 5: One-Hot Encoding ---
if len(low_card_cols) > 0:
    print(f"\n⏳ [5/6] Đang thực hiện One-Hot Encoding...")
    df_merged = pd.get_dummies(df_merged, columns=low_card_cols, dummy_na=True)

print(f"   ✅ Tổng số cột sau cùng: {df_merged.shape[1]}")
df_merged = reduce_mem_usage(df_merged)

# --- BƯỚC 6: Lưu file ---
print(f"\n⏳ [6/6] Đang lưu file kết quả vào {OUTPUT_FILE}...")
df_merged.to_csv(OUTPUT_FILE, index=False)
print("\n🎉 HOÀN TẤT! Dữ liệu đã được lọc theo danh sách uy tín và xử lý tối ưu.")