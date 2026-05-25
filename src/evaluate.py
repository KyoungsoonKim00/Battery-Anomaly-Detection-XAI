# src/evaluate.py
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from data_loader import get_dataloader
from model import TransformerAutoencoder

# 1. Config 로드
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def get_test_labels(data_dir, test_files, window_size):
    """테스트 파일명에 'OK'가 있으면 0(정상), 'NG'가 있으면 1(불량)로 라벨링"""
    labels = []
    for file_name in test_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # 슬라이딩 윈도우로 생성될 샘플 수 계산
            num_samples = len(df) - window_size
            label = 0 if "OK" in file_name.upper() else 1
            labels.extend([label] * num_samples)
    return np.array(labels)

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() and config['env']['device'] == 'auto' else "cpu")
    print(f"🚀 Evaluating on device: {device}")
    
    # 2. 데이터 및 라벨 로드 (test 모드에서는 shuffle=False 여야 라벨과 매핑됨)
    test_loader, _ = get_dataloader(mode="test")
    y_true = get_test_labels(
        config['data']['sample_dir'], 
        config['data']['test_files'], 
        config['data']['window_size']
    )
    
    # 3. 모델 및 임계값 로드
    model = TransformerAutoencoder(
        input_dim=208, 
        window_size=config['data']['window_size']
    ).to(device)
    
    model_path = "models/best_model.pth"
    thresh_path = "models/threshold.txt"
    
    if not os.path.exists(model_path) or not os.path.exists(thresh_path):
        print("⚠️ [Error] 학습된 모델이나 임계값 파일이 없습니다. train.py를 먼저 실행하세요.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with open(thresh_path, "r") as f:
        threshold = float(f.read().strip())
        
    criterion = nn.MSELoss(reduction='none')
    y_scores = []
    
    # 4. 추론(Inference) 진행
    print("통계적 Anomaly Score 계산 중...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            mse = criterion(reconstructed, batch).mean(dim=[1, 2]).cpu().numpy()
            y_scores.extend(mse)
            
    y_scores = np.array(y_scores)
    
    # 길이가 혹시 맞지 않으면 최소 길이로 맞춤 (배치 드롭 등 예외 처리)
    min_len = min(len(y_true), len(y_scores))
    y_true, y_scores = y_true[:min_len], y_scores[:min_len]
    
    # 5. 정량적 지표 계산
    y_pred = (y_scores > threshold).astype(int)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print("📊 [최종 테스트 성능 지표 (Test Set)]")
    print("="*50)
    print(f" - Precision (정밀도) : {precision:.4f}")
    print(f" - Recall (재현율)    : {recall:.4f}")
    print(f" - F1-Score         : {f1:.4f}")
    print(f" - 최적 임계값(Threshold) : {threshold:.4f}")
    print("\n [Confusion Matrix]")
    print(f" True OK(0) | {cm[0][0]:>5} (TN) | {cm[0][1]:>5} (FP: 과검)")
    print(f" True NG(1) | {cm[1][0]:>5} (FN: 미검) | {cm[1][1]:>5} (TP)")
    print("="*50)
    
    # 6. 결과 시각화 및 이미지 저장 (README 연동용)
    os.makedirs("docs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # 정상/불량 스코어 분리
    normal_scores = y_scores[y_true == 0]
    anomaly_scores = y_scores[y_true == 1]
    
    # 히스토그램 플롯
    plt.hist(normal_scores, bins=50, alpha=0.6, color='blue', label='Normal (OK)', density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.6, color='red', label='Anomaly (NG)', density=True)
    
    # 임계값 수직선
    plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.4f})')
    
    plt.title("Distribution of Reconstruction Errors (Anomaly Scores)")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "docs/anomaly_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📸 시각화 이미지가 성공적으로 저장되었습니다: {save_path}")
    print("README.md에 이 이미지를 삽입하여 수학적 변별력을 증명하세요!")

if __name__ == "__main__":
    evaluate_model()