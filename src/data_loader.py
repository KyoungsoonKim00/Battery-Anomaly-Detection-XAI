# src/data_loader.py
import os
import yaml
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib # 스케일러 저장을 위해 추가

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class BatteryDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size]
        return torch.tensor(window, dtype=torch.float32)

def load_multiple_files(file_list, data_dir):
    """여러 CSV 파일을 읽어와 하나의 DataFrame으로 병합"""
    df_list = []
    drop_cols = ['Date', 'Time', 'SerialNumber']
    
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = df.drop(columns=[col for col in drop_cols if col in df.columns])
            df_list.append(df)
        else:
            print(f"Warning: {file_name} not found in {data_dir}")
            
    return pd.concat(df_list, axis=0, ignore_index=True) if df_list else pd.DataFrame()

def get_dataloader(mode="train", scaler=None):
    """mode('train', 'val', 'test')에 따라 적절한 DataLoader 반환"""
    data_dir = config['data']['sample_dir']
    
    # mode에 따라 config에서 읽어올 파일 리스트 결정
    if mode == "train":
        file_list = config['data']['train_files']
    elif mode == "val":
        file_list = config['data']['val_files']
    else:
        file_list = config['data']['test_files']
        
    df = load_multiple_files(file_list, data_dir)
    
    # Train 모드일 때만 Scaler를 Fit(학습)하고 저장, 나머지는 Transform(적용)만 수행 (Data Leakage 방지)
    if mode == "train":
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.values)
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl") # 학습된 스케일러 저장
    else:
        if scaler is None:
            scaler = joblib.load("models/scaler.pkl")
        scaled_data = scaler.transform(df.values)
        
    dataset = BatteryDataset(scaled_data, config['data']['window_size'])
    # 학습 시에만 데이터를 섞음(Shuffle)
    shuffle = True if mode == "train" else False 
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=shuffle)
    
    return dataloader, scaler