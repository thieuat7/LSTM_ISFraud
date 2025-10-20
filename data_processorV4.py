import pandas as pd
import numpy as np
import joblib
import subprocess
import sys
import os
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from datetime import datetime

class DataProcessorV4:
    def __init__(self, seq_len=20, dayfirst=False, model_dir="./preprocessors", train_split_ratio=0.8):
        self.seq_len = seq_len
        self.dayfirst = dayfirst
        self.model_dir = model_dir
        self.train_split_ratio = train_split_ratio
        self.input_dim = None

        # Khởi tạo tất cả preprocessors
        self.user_encoder = None
        self.receiver_encoder = None
        self.scaler = None
        self.cat_encoder = None
        self.feature_means = {}
        self.timestamp_median = None

        # ## <-- CẬP NHẬT: Tái cấu trúc lại logic save/load dùng dictionary
        # Quản lý tất cả các đối tượng cần lưu ở một nơi duy nhất
        self.processors_to_save = {
            'user_encoder': 'user_encoder.pkl',
            'receiver_encoder': 'receiver_encoder.pkl',
            'scaler': 'scaler.pkl',
            'cat_encoder': 'cat_encoder.pkl',
            'feature_means': 'feature_means.pkl',
            'timestamp_median': 'timestamp_median.pkl'
        }

        self.numeric_features = [
            "transaction_amount", "account_balance_before_txn", "avg_sent_amount_prev",
            "avg_received_amount_prev", "deviation_from_avg_sent_amount",
            "deviation_from_avg_received_amount", "transaction_count_last_7d",
            "unix_time", "has_sent_before", "has_received_before",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos"
        ]
        self.categorical_features = ["transaction_type"]

    def _create_feature_windows(self, df_subset, all_features):
        """Tạo cửa sổ trượt cho các features (X)."""
        windows = []
        user_groups = df_subset.groupby("user_id_encoded")
        
        for _, group in user_groups:
            idx = group.index.to_numpy()
            user_features = all_features[idx]
            
            if len(user_features) < self.seq_len:
                pad_len = self.seq_len - len(user_features)
                padded = np.pad(user_features, ((pad_len, 0), (0, 0)), mode='constant', constant_values=0)
                windows.append(padded)
            else:
                for i in range(len(user_features) - self.seq_len + 1):
                    windows.append(user_features[i:i + self.seq_len])
        return np.array(windows, dtype=np.float32)

    def _create_label_windows(self, df_subset, all_labels):
        """Tạo nhãn cho mỗi cửa sổ (chỉ lấy nhãn của item cuối cùng)."""
        labels = []
        user_groups = df_subset.groupby("user_id_encoded")
        
        for _, group in user_groups:
            idx = group.index.to_numpy()
            user_labels = all_labels[idx]
            
            # Chỉ tạo nhãn cho các cửa sổ đủ dài (không padding)
            if len(user_labels) >= self.seq_len:
                for i in range(len(user_labels) - self.seq_len + 1):
                    # Lấy nhãn của item cuối cùng trong cửa sổ
                    labels.append(user_labels[i + self.seq_len - 1])
        return np.array(labels, dtype=np.float32)

    def _preprocess_dataframe(self, df):
        """Hàm phụ trợ để thực hiện các bước tiền xử lý chung."""
        df = df.copy()
        
        # 1. Xử lý timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=self.dayfirst, errors="coerce")
        df["timestamp"] = df["timestamp"].fillna(self.timestamp_median)
        df["unix_time"] = df["timestamp"].astype("int64") // 10**9
        df["time_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["time_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["time_of_day"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        # 2. Xử lý NaN cho categorical
        df[self.categorical_features] = df[self.categorical_features].fillna("missing")

        # 3. Ép kiểu và mã hóa ID
        df["user_id"] = df["user_id"].astype(str)
        df["receiver_id"] = df["receiver_id"].astype(str)
        df["user_id_encoded"] = self.user_encoder.transform(df[["user_id"]]).astype(int)
        df["receiver_id_encoded"] = self.receiver_encoder.transform(df[["receiver_id"]]).astype(int)
        
        return df.sort_values(["user_id_encoded", "timestamp"]).reset_index(drop=True)

    def fit_transform(self, csv_path, label_col=None):
        df = pd.read_csv(csv_path, encoding="latin-1")
        df = df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(df) * self.train_split_ratio)
        train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

        # Fit encoders và tính toán các giá trị CHỈ từ tập train
        self.timestamp_median = pd.to_datetime(train_df["timestamp"], dayfirst=self.dayfirst, errors="coerce").median()
        self.user_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(train_df[["user_id"]].astype(str))
        self.receiver_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(train_df[["receiver_id"]].astype(str))
        
        # Áp dụng hàm tiền xử lý chung
        train_df = self._preprocess_dataframe(train_df)
        test_df = self._preprocess_dataframe(test_df)

        # Fill NaN numeric, fit scaler
        for col in self.numeric_features:
            mean_val = train_df[col].mean()
            self.feature_means[col] = mean_val
            train_df[col] = train_df[col].fillna(mean_val)
            test_df[col] = test_df[col].fillna(mean_val)
        
        self.scaler = MinMaxScaler().fit(train_df[self.numeric_features])
        X_train_num = self.scaler.transform(train_df[self.numeric_features])
        X_test_num = self.scaler.transform(test_df[self.numeric_features])

        # Fit OneHotEncoder
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train_df[self.categorical_features])
        X_train_cat = self.cat_encoder.transform(train_df[self.categorical_features])
        X_test_cat = self.cat_encoder.transform(test_df[self.categorical_features])

        # Gộp features
        X_train_all = np.concatenate([X_train_num, X_train_cat], axis=1)
        X_test_all = np.concatenate([X_test_num, X_test_cat], axis=1)
        self.input_dim = X_train_all.shape[1]

        # Tạo cửa sổ
        X_train = self._create_feature_windows(train_df, X_train_all)
        X_test = self._create_feature_windows(test_df, X_test_all)
        print(f"✅ Created {len(X_train)} training windows and {len(X_test)} testing windows.")

        if label_col:
            y_train_full = train_df[label_col].values
            y_test_full = test_df[label_col].values
            # ## <-- CẬP NHẬT: Dùng hàm tạo label riêng
            y_train = self._create_label_windows(train_df, y_train_full)
            y_test = self._create_label_windows(test_df, y_test_full)
            print("✅ Data processing complete (supervised).")
            return X_train, X_test, y_train, y_test, self.input_dim
        else:
            print("✅ Data processing complete (unsupervised).")
            return X_train, X_test, self.input_dim

    def transform_new_data(self, df_new, version_name="latest"):
        if self.user_encoder is None:
            print(f"Processors not loaded. Loading version '{version_name}'...")
            self.load_processors(version_name)
        
        # Áp dụng pipeline xử lý hoàn chỉnh
        df_processed = self._preprocess_dataframe(df_new)
        
        for col in self.numeric_features:
            df_processed[col] = df_processed[col].fillna(self.feature_means.get(col, 0))

        X_num = self.scaler.transform(df_processed[self.numeric_features])
        X_cat = self.cat_encoder.transform(df_processed[self.categorical_features])
        X_all = np.concatenate([X_num, X_cat], axis=1)

        windows = self._create_feature_windows(df_processed, X_all)
        return windows

    # ## <-- CẬP NHẬT: Mang logic save/load linh hoạt trở lại
    def save_processors(self, version_name=None):
        """
        Lưu các processor vào một thư mục phiên bản và tạo một liên kết 'latest'
        để trỏ đến phiên bản mới nhất.
        """
        if version_name is None:
            version_name = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        version_path = os.path.join(self.model_dir, version_name)
        os.makedirs(version_path, exist_ok=True)
        
        # Lưu các file processor
        for attr_name, filename in self.processors_to_save.items():
            processor = getattr(self, attr_name)
            if processor is not None:
                joblib.dump(processor, os.path.join(version_path, filename))
        
        print(f"✅ Đã lưu các processor cho phiên bản '{version_name}' vào {version_path}")
        
        # --- PHẦN CẬP NHẬT ---
        # Tạo liên kết 'latest' để dễ dàng truy cập phiên bản mới nhất
        latest_path = os.path.join(self.model_dir, "latest")
        
        # Xóa link cũ nếu tồn tại (cách này hoạt động cho cả symlink và junction)
        if os.path.lexists(latest_path):
            # Trên Windows, junction được xem như thư mục, cần dùng rmdir
            if sys.platform == "win32" and os.path.isdir(latest_path):
                os.rmdir(latest_path)
            else:
                os.remove(latest_path)

        try:
            if sys.platform == "win32":
                # Trên Windows: Dùng mklink /J để tạo Directory Junction (không cần admin)
                # Lệnh: mklink /J <Đường dẫn Link> <Đường dẫn Thư mục gốc>
                # Quan trọng: Dùng version_path (đường dẫn đầy đủ)
                subprocess.run(
                    ['cmd', '/c', 'mklink', '/J', latest_path, version_path], 
                    check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            else:
                # Trên Linux/macOS: Dùng os.symlink như bình thường.
                # Dùng version_name (đường dẫn tương đối) sẽ linh hoạt hơn
                os.symlink(version_name, latest_path, target_is_directory=True)
            
            print(f"✅ Đã tạo/cập nhật liên kết 'latest' -> '{version_name}'")

        except (subprocess.CalledProcessError, OSError) as e:
            print(f"⚠️  Không thể tạo liên kết 'latest': {e}")
            print("    -> Trên Windows, hãy thử bật 'Chế độ nhà phát triển' hoặc chạy lại với quyền Admin.")
        # --- KẾT THÚC PHẦN CẬP NHẬT ---
    
        return version_name

    def load_processors(self, version_name="latest"):
        version_path = os.path.join(self.model_dir, version_name)
        if not os.path.isdir(version_path):
            print(f"❌ Error: Version directory not found at {version_path}")
            return
            
        try:
            for attr_name, filename in self.processors_to_save.items():
                file_path = os.path.join(version_path, filename)
                setattr(self, attr_name, joblib.load(file_path))
            print(f"✅ Loaded processors from version '{version_name}'")
        except FileNotFoundError as e:
            print(f"❌ Error: Failed to load. File not found: {e.filename}")