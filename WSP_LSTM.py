import os
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    data_mast = pd.read_csv(file_path, sep=';')
    # 选择特征列
    features = [
        'WS1_035_Mean', 'Baro_035_Mean', 
        'Hum_035_Mean', 'Rain_Mean', 'Temp_035_Mean'
    ]
    dataset = data_mast[features].values.astype(np.float32)
    time = pd.to_datetime(data_mast['date'] + " " + data_mast['time'])
    return dataset, time

# 数据标准化
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler

# 滑动窗口生成（向量化实现）
def create_dataset(data, window_size, predict_steps):
    num_samples = data.shape[0] - window_size - predict_steps + 1
    indices = np.arange(window_size)[None, :] + np.arange(num_samples)[:, None]
    X = data[indices]
    Y = data[indices + window_size + np.arange(predict_steps)[None, :], 0]
    return X, Y

def create_data_pipeline(X, Y, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# 反归一化处理
def inverse_transform(predictions, scaler):
    dummy = np.zeros((predictions.size, 5))
    dummy[:, 0] = predictions.flatten()
    return scaler.inverse_transform(dummy)[:, 0].reshape(predictions.shape)

# ========================
# 模型架构优化
# ========================
def build_model():
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(N, 5)),  # 第一层LSTM
        LSTM(64, return_sequences=False),                       # 第二层LSTM
        Dense(32, activation='relu'),                          # 中间全连接层
        Dense(M, dtype='float32')                              # 输出层保持FP32精度
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model

if __name__ == "__main__":
    # ========================
    # GPU配置与显存优化
    # ========================
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置显存按需增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} 物理GPU, {len(logical_gpus)} 逻辑GPU")
        except RuntimeError as e:
            print(e)
    # 启用混合精度训练
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # ========================
    # 数据加载与预处理优化
    # ========================
    # 数据路径
    fold_dir = os.getcwd()
    #fold_dir = r'C:\Users\39736\OneDrive\专利\专利代码'
    file_name = "/Mast_2023-06-06_00-00-00_2024-05-06_00-00-00.csv"
    file_path = os.path.join(fold_dir, file_name.lstrip('/'))
    # 加载数据

    # 数据准备流程
    dataset, time_series = load_data()
    #print(dataset, time_series)
    data_scaled, scaler = normalize_data(dataset)

    # 定义窗口参数
    N = 144 * 15  # 历史窗口长度（15天）
    for j in [1, 4, 12, 24, 36, 72]:
        M = 6 * j    # 预测长度（j小时）
        # 生成数据集
        #X, Y = create_dataset(data_scaled, N, M)
        #time_windows = np.array([time_series[i+N:i+N+M] for i in range(len(data_scaled)-N-M+1)])
        X, Y, T = [], [], []
        for i in range(len(data_scaled)-N-M+1):
            X.append(data_scaled[i:i+N])
            Y.append(data_scaled[i+N:i+N+M,0])
            T.append(time_series[i+N:i+N+M])
        X = np.array(X)
        Y = np.array(Y)
        T = np.array(T)
        # 数据集划分
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size], Y[train_size:]
        time_test = T[train_size:]
        # ========================
        # 数据管道优化
        # ========================
        batch_size = 256  # 根据GPU显存调整
        buffer_size = 1024

        train_dataset = create_data_pipeline(X_train, Y_train, shuffle=True)
        test_dataset = create_data_pipeline(X_test, Y_test)

        model = build_model()
        model.summary()

        # ========================
        # 训练过程优化
        # ========================
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True),
            # 保存模型
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/wind_speed_model_{}.keras'.format(int(M//6)),
                save_best_only=True,
                monitor='val_mae'),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5)
        ]

        history = model.fit(
            train_dataset,
            epochs=50,
            validation_data=test_dataset,
            callbacks=callbacks,
            verbose=1
        )

        # ========================
        # 预测与结果可视化
        # ========================
        # 加载最佳模型
        model = tf.keras.models.load_model('models/wind_speed_model_{}.keras'.format(int(M//6)))

        # 批量预测
        predictions = model.predict(X_test, batch_size=1024)        

        Y_pred = inverse_transform(predictions, scaler)
        Y_true = inverse_transform(Y_test, scaler)
        # 保存归一化参数（在训练结束后执行）
        
        dump(scaler, 'models/minmax_scaler_{}.joblib'.format(int(M//6)))

# 保存模型参数（已通过ModelCheckpoint自动保存，无需额外操作）

        # 评估指标
        rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
        mae = mean_absolute_error(Y_true, Y_pred)
        print(f"测试集RMSE: {rmse:.3f}")
        print(f"测试集MAE: {mae:.3f}")
        # 展平数据
        time_flat = time_test.flatten()
        Y_true_flat = Y_true.flatten()
        Y_pred_flat = Y_pred.flatten()
        
        # 可视化
        plt.figure(figsize=(15, 6))
        plt.scatter(time_flat, Y_true_flat, c='blue', marker='+', label='真实值', alpha=0.6)
        plt.scatter(time_flat, Y_pred_flat, c='red', marker='.', label='预测值', alpha=0.6)
        plt.text(0.3, 0.9, f'RMSE: {rmse:.2f} m/s\nMAE: {mae:.3f} m/s', 
                transform=plt.gca().transAxes, fontsize=18, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
        plt.title('风速预测{:.0f}小时结果对比'.format(M//6))
        plt.xlabel('时间', fontsize=20)
        plt.ylabel('风速 (m/s)', fontsize=20)
        plt.legend(fontsize=18)  
        plt.grid(True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        # 坐标轴字体大小
        plt.tick_params(axis='both', which='major', labelsize=15)
        #plt.tick_params(axis='both', which='minor', labelsize=15)   
        plt.savefig(f'fig/wind_speed_prediction_{M//6}.png', dpi=1000, bbox_inches='tight')
        plt.tight_layout()
        # 显存清理
        del X_test, Y_test
        tf.keras.backend.clear_session()
    #plt.show()

    
