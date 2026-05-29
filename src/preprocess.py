import os
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import config

def load_and_preprocess_train_data():
    train_files = []
    for i in range(1000, 1051):
        train_files.append(f'{i}_chg.csv')
        train_files.append(f'{i}_dchg.csv')

    all_train_df = []
    for f in train_files:
        file_path = os.path.join(config.TRAIN_DIR, f)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, dtype=object)
            df = df[config.SENSOR_COLS]
            all_train_df.append(df)
            
    if not all_train_df:
        raise ValueError("학습 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요.")

    train_data_full = pd.concat(all_train_df, ignore_index=True)
    train_data_full = train_data_full.apply(pd.to_numeric, errors='coerce')
    train_data_full.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputer = SimpleImputer(strategy='constant', fill_value=0.0)
    train_data_imputed = imputer.fit_transform(train_data_full)
    train_data_full = pd.DataFrame(train_data_imputed, columns=config.SENSOR_COLS)

    imputer_save_path = os.path.join(config.TRAIN_DIR, "imputer.pkl")
    with open(imputer_save_path, 'wb') as f:
        pickle.dump(imputer, f)

    sensor_stats = train_data_full.describe().transpose()
    constant_sensors = sensor_stats[sensor_stats['std'] < 1e-9].index.tolist()
    final_sensor_cols = [col for col in config.SENSOR_COLS if col not in constant_sensors]

    final_cols_save_path = os.path.join(config.TRAIN_DIR, "final_sensor_cols.pkl")
    with open(final_cols_save_path, 'wb') as f:
        pickle.dump(final_sensor_cols, f)

    train_data_filtered = train_data_full[final_sensor_cols]
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data_filtered)

    scaler_save_path = os.path.join(config.TRAIN_DIR, "transformer_scaler.pkl")
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)

    return train_data_scaled, final_sensor_cols