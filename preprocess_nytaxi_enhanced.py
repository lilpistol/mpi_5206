import os
import json
import pickle
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _clip_series(s: pd.Series, low=None, high=None) -> pd.Series:
    if low is not None:
        s = s.clip(lower=low)
    if high is not None:
        s = s.clip(upper=high)
    return s


def _build_topk_ohe(series: pd.Series, top_values: List[int], prefix: str) -> pd.DataFrame:
    # Map values to columns for top-k, else to 'Other'
    cols = {f"{prefix}_{val}": (series == val).astype(np.float32) for val in top_values}
    other_mask = ~series.isin(top_values)
    cols[f"{prefix}_Other"] = other_mask.astype(np.float32)
    return pd.DataFrame(cols, index=series.index)


def preprocess_nytaxi_enhanced(
    csv_path: str = "nytaxi2022.csv",
    out_dir: str = "data_preprocessed_enhanced",
    test_size: float = 0.3,
    random_state: int = 42,
    sample_frac: float = 0.1,
    top_k_pu: int = 100,
    top_k_do: int = 100,
    y_min: float = 0.0,
    y_max: float = 1000.0,
    stratify_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Enhanced preprocessing with feature engineering to improve model expressiveness.

    - Time features: hour, day-of-week, weekend, cyclic encoding (sin/cos).
    - Numerical features: duration(min), trip_distance, passenger_count, extra, avg_speed.
    - Categorical encoding:
        * RatecodeID: one-hot using training categories (+Other for unseen)
        * payment_type: one-hot using training categories (+Other for unseen)
        * PULocationID/DOLocationID: top-K one-hot + Other (based on training frequency)
        * is_same_zone (PU==DO)
    - Standardization: fit StandardScaler on X_train, apply to X_test.

    Saves to `out_dir`:
      - X_train_scaled.npy, X_test_scaled.npy, y_train.npy, y_test.npy, scaler.pkl, feature_meta.json
    """
    os.makedirs(out_dir, exist_ok=True)

    print("1) 读取 CSV ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"原始数据形状: {df.shape}")
    if sample_frac and 0.0 < sample_frac < 1.0:
        print(f"采样 {sample_frac*100:.1f}% 数据用于预处理与训练……")
        df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"采样后数据形状: {df.shape}")

    print("2) 时间解析与行程时长 ...")
    df['pickup'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df['dropoff'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df['duration'] = (df['dropoff'] - df['pickup']).dt.total_seconds() / 60.0

    print("3) 基础特征与标签选择 ...")
    cols_needed = [
        'duration', 'passenger_count', 'trip_distance', 'RatecodeID',
        'PULocationID', 'DOLocationID', 'payment_type', 'extra',
        'pickup'
    ]
    target_col = 'total_amount'
    df = df[cols_needed + [target_col]].copy()

    print("4) 缺失与基础质量控制 ...")
    # Drop NA in required columns
    df = df.dropna(subset=cols_needed + [target_col]).copy()

    # 过滤异常标签：y <= y_min 或 y > y_max 的样本剔除
    before = df.shape[0]
    df = df[(df[target_col] > y_min) & (df[target_col] <= y_max)].copy()
    after = df.shape[0]
    print(f"4.1) 标签过滤: 移除 {before - after} 行 (y <= {y_min} 或 y > {y_max})，剩余 {after}")

    # Clip outliers to reasonable ranges
    df['trip_distance'] = _clip_series(df['trip_distance'], 0.0, 50.0)
    df['duration'] = _clip_series(df['duration'], 0.1, 180.0)  # 0.1 min to 3 hours
    df['passenger_count'] = _clip_series(df['passenger_count'], 0, 6)
    df['extra'] = _clip_series(df['extra'], -5.0, 10.0)

    print("5) 时间特征 ...")
    df['hour'] = df['pickup'].dt.hour.astype(np.int16)
    df['dow'] = df['pickup'].dt.dayofweek.astype(np.int16)  # Monday=0
    df['is_weekend'] = (df['dow'] >= 5).astype(np.int8)
    # Cyclic encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7.0)

    print("6) 派生数值特征 ...")
    # Avg speed in miles/hour, avoid div by zero
    df['avg_speed'] = (df['trip_distance'] / (df['duration'] / 60.0)).replace([np.inf, -np.inf], np.nan)
    df['avg_speed'] = df['avg_speed'].fillna(0.0)
    df['avg_speed'] = _clip_series(df['avg_speed'], 0.0, 80.0)
    df['is_same_zone'] = (df['PULocationID'] == df['DOLocationID']).astype(np.int8)

    print("7) 训练/测试拆分（分层按 y 桶，避免尾部分布失衡） ...")
    base_cols = [
        'duration', 'passenger_count', 'trip_distance', 'extra',
        'avg_speed', 'is_same_zone', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'RatecodeID', 'payment_type', 'PULocationID', 'DOLocationID'
    ]
    df = df[base_cols + [target_col]].copy()

    # 构造 y 分层桶（≤50、50–100、100–200、>200 up to y_max）
    if stratify_y:
        bins = [y_min, 50.0, 100.0, 200.0, 1000,2000,2000.1]
        labels = ["<=50", "50-100", "100-200", "200-1000","1000-2000",">2000"]
        df['y_bucket'] = pd.cut(df[target_col], bins=bins, labels=labels, include_lowest=True, right=True)
        # 若有边界导致的 NA，直接删除
        df = df.dropna(subset=['y_bucket'])
        # 打印桶占比
        dist = df['y_bucket'].value_counts(normalize=True).sort_index()
        print("y 桶占比：", dist.to_dict())
        # 分层拆分
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['y_bucket']
        )
        df_train = df_train.drop(columns=['y_bucket'])
        df_test = df_test.drop(columns=['y_bucket'])
    else:
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"训练集: {df_train.shape}, 测试集: {df_test.shape}")

    # Mapping based on training data
    print("8) 基于训练集统计类别映射 ...")
    ratecode_vals = df_train['RatecodeID'].astype(int).value_counts().index.tolist()
    payment_vals = df_train['payment_type'].astype(int).value_counts().index.tolist()
    # Top-K for PU/DO
    pu_top = df_train['PULocationID'].astype(int).value_counts().index.tolist()[:top_k_pu]
    do_top = df_train['DOLocationID'].astype(int).value_counts().index.tolist()[:top_k_do]

    def _encode_block(block: pd.DataFrame) -> pd.DataFrame:
        # Numeric
        num = block[['duration', 'passenger_count', 'trip_distance', 'extra', 'avg_speed']].astype(np.float32)
        time_num = block[['is_same_zone', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']].astype(np.float32)

        # One-hot for low-card categories with 'Other'
        rc = block['RatecodeID'].astype(int)
        rc = rc.where(rc.isin(ratecode_vals), other=-1)
        rc_df_list = []
        for v in ratecode_vals:
            rc_df_list.append((rc == v).astype(np.float32).rename(f"RatecodeID_{v}"))
        rc_df_list.append((rc == -1).astype(np.float32).rename("RatecodeID_Other"))
        rc_ohe = pd.concat(rc_df_list, axis=1)

        pay = block['payment_type'].astype(int)
        pay = pay.where(pay.isin(payment_vals), other=-1)
        pay_df_list = []
        for v in payment_vals:
            pay_df_list.append((pay == v).astype(np.float32).rename(f"payment_type_{v}"))
        pay_df_list.append((pay == -1).astype(np.float32).rename("payment_type_Other"))
        pay_ohe = pd.concat(pay_df_list, axis=1)

        # Top-K OHE for PU/DO with Other
        pu = block['PULocationID'].astype(int)
        pu_ohe = _build_topk_ohe(pu, pu_top, prefix="PU")
        do = block['DOLocationID'].astype(int)
        do_ohe = _build_topk_ohe(do, do_top, prefix="DO")

        # Concatenate
        X_df = pd.concat([num, time_num, rc_ohe, pay_ohe, pu_ohe, do_ohe], axis=1)
        X_df = X_df.fillna(0.0)
        return X_df

    print("9) 构建特征矩阵 ...")
    X_train_df = _encode_block(df_train)
    X_test_df = _encode_block(df_test)
    y_train = df_train[target_col].astype(np.float32).values
    y_test = df_test[target_col].astype(np.float32).values

    feature_names = X_train_df.columns.tolist()
    print(f"特征维度: {len(feature_names)}")

    print("10) 特征标准化 ...")
    scaler = StandardScaler()
    # 为降低内存使用，将结果转为 float32 保存
    X_train_scaled = scaler.fit_transform(X_train_df.values).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_df.values).astype(np.float32)

    print("11) 保存输出 ...")
    np.save(os.path.join(out_dir, 'X_train_scaled.npy'), X_train_scaled)
    np.save(os.path.join(out_dir, 'X_test_scaled.npy'), X_test_scaled)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)

    with open(os.path.join(out_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    meta = {
        'top_k_pu': top_k_pu,
        'top_k_do': top_k_do,
        'pu_top_values': pu_top,
        'do_top_values': do_top,
        'ratecode_values': ratecode_vals,
        'payment_values': payment_vals,
        'feature_names': feature_names,
        'random_state': random_state,
        'test_size': test_size,
        'y_min': y_min,
        'y_max': y_max,
        'stratify_y': stratify_y,
        'y_bins': [y_min, 50.0, 100.0, 200.0, y_max],
    }
    with open(os.path.join(out_dir, 'feature_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("完成！输出目录:", out_dir)
    return X_train_scaled, X_test_scaled, y_train, y_test


def parse_args():
    p = argparse.ArgumentParser(description="Enhanced NYTaxi preprocessing with sampling and OHE/time features")
    p.add_argument("--csv", default="nytaxi2022.csv", help="Path to CSV file")
    p.add_argument("--out-dir", default="data_preprocessed_enhanced", help="Output directory")
    p.add_argument("--test-size", type=float, default=0.3, help="Test split fraction")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--sample-frac", type=float, default=0.1, help="Sampling fraction (0<frac<1); use 1.0 for full data")
    p.add_argument("--top-k-pu", type=int, default=100, help="Top-K PULocationID for OHE (rest to Other)")
    p.add_argument("--top-k-do", type=int, default=100, help="Top-K DOLocationID for OHE (rest to Other)")
    p.add_argument("--y-min", type=float, default=0.0, help="Filter labels: keep y > y_min")
    p.add_argument("--y-max", type=float, default=1000.0, help="Filter labels: keep y <= y_max")
    p.add_argument("--no-stratify-y", action="store_true", help="Disable stratification by y buckets for train/test split")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess_nytaxi_enhanced(
        csv_path=args.csv,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        sample_frac=args.sample_frac,
        top_k_pu=args.top_k_pu,
        top_k_do=args.top_k_do,
        y_min=args.y_min,
        y_max=args.y_max,
        stratify_y=(not args.no_stratify_y),
    )
