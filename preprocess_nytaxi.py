import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_nytaxi_data(csv_path='nytaxi2022.csv'):
    """
    数据预处理：把原始 CSV 变成能喂进 NumPy 网络里的干净矩阵

    Returns:
        X_train_scaled: 标准化后的训练特征 (ndarray)
        X_test_scaled: 标准化后的测试特征 (ndarray)
        y_train: 训练标签 (ndarray)
        y_test: 测试标签 (ndarray)
    """

    dtype_dict = {
        'VendorID': 'int8',
        'passenger_count': 'int8',
        'RatecodeID': 'int8',
        'store_and_fwd_flag': 'str',
        'PULocationID': 'int16',
        'DOLocationID': 'int16',
        'payment_type': 'int8',
        'fare_amount': 'float32',
        'extra': 'float32',
        'mta_tax': 'float32',
        'tip_amount': 'float32',
        'tolls_amount': 'float32',
        'improvement_surcharge': 'float32',
        'total_amount': 'float32',
        'congestion_surcharge': 'float32',
        'airport_fee': 'float32'
    }

    df = pd.read_csv(csv_path, dtype=dtype_dict, low_memory=False)
    print(f"原始数据形状: {df.shape}")
    
    print("2. 计算行程时长...")
    # ② 美式 12 小时制 → datetime
    df['pickup'] = pd.to_datetime(df['tpep_pickup_datetime'],
                                  format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df['dropoff'] = pd.to_datetime(df['tpep_dropoff_datetime'],
                                   format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df['duration'] = (df['dropoff'] - df['pickup']).dt.total_seconds() / 60


    print("3. 选择特征和标签列...")
    # 特征列：duration + passenger_count + trip_distance + RatecodeID + PULocationID + DOLocationID + payment_type + extra
    feature_columns = ['duration', 'passenger_count', 'trip_distance', 'RatecodeID',
                      'PULocationID', 'DOLocationID', 'payment_type', 'extra']

    # 标签列：total_amount
    target_column = 'total_amount'

    # 提取特征和标签
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    print(f"特征形状: {X.shape}")
    print(f"标签形状: {y.shape}")

    print("4. 处理缺失值...")
    # 合并X和y，然后删除包含缺失值的行
    df_clean = pd.concat([X, y], axis=1).dropna()

    # 重新分离特征和标签
    X_clean = df_clean[feature_columns].values
    y_clean = df_clean[target_column].values

    print(f"清洗后数据形状: X={X_clean.shape}, y={y_clean.shape}")

    print("5. 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=42
    )

    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

    print("6. 标准化特征...")
    # 只在训练集上拟合StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # y标签保持原样不动

    print("7. 数据预处理完成！")
    print(f"最终数据形状:")
    print(f"  X_train_scaled: {X_train_scaled.shape}")
    print(f"  X_test_scaled: {X_test_scaled.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")

    # 保存标准化器，方便后续使用
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # 执行数据预处理
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_nytaxi_data()

    # 保存处理后的数据为numpy数组文件
    np.save('X_train_scaled.npy', X_train_scaled)
    np.save('X_test_scaled.npy', X_test_scaled)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    print("\n数据已保存为numpy文件:")
    print("- X_train_scaled.npy")
    print("- X_test_scaled.npy")
    print("- y_train.npy")
    print("- y_test.npy")
    print("- scaler.pkl (标准化器)")