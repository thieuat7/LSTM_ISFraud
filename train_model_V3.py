import joblib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, LayerNormalization, SpatialDropout1D# type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

class LSTMAutoencoderTrainer:
    """
    Class để huấn luyện một mô hình LSTM Autoencoder và tính toán ngưỡng.
    """
    def __init__(self, seq_len: int = 5, input_dim: int = 5, latent_dim: int = 64):
        """
        Khởi tạo class trainer cho LSTM Autoencoder.

        Args:
            seq_len (int): Chiều dài chuỗi.
            input_dim (int): Số lượng features.
            latent_dim (int): Số lượng neuron trong LSTM encoder/decoder.
        """
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.threshold = None
        self.model = self._build_model()

    def _build_model(self) -> Model:
        """
        Xây dựng kiến trúc mô hình Stacked LSTM Autoencoder (phiên bản tối ưu).
        """
        # Encoder
        inputs = Input(shape=(self.seq_len, self.input_dim))

        # LSTM encoder layer 1
        x = LSTM(self.latent_dim, return_sequences=True,
                dropout=0.2,
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name='encoder_lstm_1')(inputs)
        x = SpatialDropout1D(0.2)(x)
        x = LayerNormalization()(x)

        # LSTM encoder layer 2 (chỉ lấy vector cuối cùng)
        encoded = LSTM(self.latent_dim, return_sequences=False,
                    dropout=0.2,
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    name='encoder_lstm_2')(x)
        encoded = LayerNormalization()(encoded)

        # Decoder
        latent_vector = RepeatVector(self.seq_len, name='latent_vector')(encoded)

        # LSTM decoder (chỉ 1 lớp)
        decoded = LSTM(self.latent_dim, return_sequences=True,
                    dropout=0.2,
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    name='decoder_lstm')(latent_vector)
        decoded = LayerNormalization()(decoded)

        # Output layer
        outputs = TimeDistributed(Dense(self.input_dim), name='output_layer')(decoded)

        # Build & compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')

        return model

    def summary(self):
        self.model.summary()

    def fit(self, X_train: np.ndarray, X_val: np.ndarray = None, 
            epochs: int = 50, batch_size: int = 128, patience: int = 5):
        """
        Huấn luyện mô hình với EarlyStopping.
        
        Args:
            X_train (np.ndarray): Dữ liệu huấn luyện.
            X_val (np.ndarray, optional): Dữ liệu validation. Mặc định là None.
            epochs (int): Số epochs tối đa.
            batch_size (int): Kích thước batch.
            patience (int): Số epochs chờ trước khi dừng sớm.
        """
        # Xác định monitor dựa trên việc có validation data hay không
        monitor_metric = 'val_loss' if X_val is not None else 'loss'
        callbacks = [
            EarlyStopping(monitor=monitor_metric, patience=patience, restore_best_weights=True)
        ]

        validation_data = (X_val, X_val) if X_val is not None else None

        history = self.model.fit(X_train, X_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=validation_data,
                                 callbacks=callbacks)

        print("Training complete.")
        return history

    def calculate_and_save_threshold(self, X_data: np.ndarray, percentile: int = 98, 
                                     path: str = "models/lstm_threshold.pkl"):
        """
        Tính toán ngưỡng từ dữ liệu và lưu vào file.
        
        Args:
            X_data (np.ndarray): Dữ liệu để tính toán ngưỡng.
            percentile (int): Ngưỡng percentile để sử dụng.
            path (str): Đường dẫn để lưu file ngưỡng.
        """
        reconstructed = self.model.predict(X_data)
        train_mse = np.mean(np.power(X_data - reconstructed, 2), axis=(1, 2))
        self.threshold = np.percentile(train_mse, percentile)
        
        # Tạo thư mục nếu chưa tồn tại và lưu threshold
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.threshold, path)
        print(f"✅ Threshold saved to {path}. Value: {self.threshold:.4f}")

    def save_model(self, path: str = "models/lstm_autoencoder.keras"):
        """
        Lưu mô hình đã huấn luyện.
        
        Args:
            path (str): Đường dẫn để lưu mô hình.
        """
        if not path.endswith(".keras"):
            path += ".keras"
        self.model.save(path)
        print(f"✅ Model saved to {path}")

    def load_model(self, path: str):
        """
        Tải mô hình từ file.
        
        Args:
            path (str): Đường dẫn đến file mô hình.
        """
        from tensorflow.keras.models import load_model # type: ignore
        self.model = load_model(path)
        print(f"✅ Model loaded from {path}")

    def load_threshold(self, path: str):
        """
        Tải ngưỡng từ file.
        
        Args:
            path (str): Đường dẫn đến file ngưỡng.
        """
        self.threshold = joblib.load(path)
        print(f"✅ Threshold loaded from {path}. Value: {self.threshold:.4f}")