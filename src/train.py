import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import config
from dataset import BatteryDataset
from model import SensorTransformerAutoencoder
from preprocess import load_and_preprocess_train_data

def main():
    print("--- 1. 학습 데이터 로드 및 전처리 ---")
    train_data_scaled, final_sensor_cols = load_and_preprocess_train_data()
    num_final_sensors = len(final_sensor_cols)
    
    train_data, val_data = train_test_split(train_data_scaled, test_size=config.VALIDATION_SPLIT, random_state=42)
    train_dataset = BatteryDataset(train_data)
    val_dataset = BatteryDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = SensorTransformerAutoencoder(num_sensors=num_final_sensors).to(config.DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.PATIENCE)
    
    model_save_path = os.path.join(config.TRAIN_DIR, "transformer_autoencoder.pth")
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("--- 2. 학습 시작 ---")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(config.DEVICE)
            reconstructed = model(batch_data)
            loss = loss_fn(reconstructed, batch_data)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_VALUE)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(config.DEVICE)
                reconstructed = model(batch_data)
                loss = loss_fn(reconstructed, batch_data)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d}/{config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Val Loss 개선됨. 모델 저장 완료.")
        else:
            epochs_no_improve += 1
            print(f"  -> Val Loss 개선 없음. (Patience: {epochs_no_improve}/{config.PATIENCE})")

        if epochs_no_improve >= config.PATIENCE:
            print(f"--- 조기 종료: {config.PATIENCE} Epoch 동안 Val Loss가 개선되지 않았습니다. ---")
            break

if __name__ == "__main__":
    main()