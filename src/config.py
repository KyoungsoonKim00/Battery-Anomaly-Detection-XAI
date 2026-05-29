import torch
import os

# 1. 경로 설정 (사용자의 드라이브 환경에 맞게 수정 필요)
BASE_PATH = "/content/drive/MyDrive/KAMP_Battery_Project"
TRAIN_DIR = os.path.join(BASE_PATH, "raw_data", "train")
TEST_DIR = os.path.join(BASE_PATH, "raw_data", "test")
LABEL_DIR = os.path.join(BASE_PATH, "preprocessed", "test")

# 2. GPU 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. 208개 센서 변수 정의
VOLTAGE_COLS = [f'M{m:02d}CV{c:02d}' for m in range(1, 17) for c in range(1, 12)]
TEMP_COLS = [f'M{m:02d}T{t:02d}' for m in range(1, 17) for t in range(1, 3)]
SENSOR_COLS = VOLTAGE_COLS + TEMP_COLS

# 4. 모델 및 학습 설정
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5
VALIDATION_SPLIT = 0.3
PATIENCE = 5
CLIP_VALUE = 1.0