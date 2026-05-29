Markdown# Battery-Anomaly-Detection-XAI
> 🏭 Hugging Face DistilBert 기반 Transformer Autoencoder와 SHAP을 활용한 배터리팩 이상탐지 및 원인 규명 파이프라인

![Award](https://img.shields.io/badge/Award-데이터분석경진대회_1위_금상-gold)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)

## 💡 비즈니스 임팩트 : "데이터를 넘어 현장의 Action으로" 
본 프로젝트는 단순한 이상탐지를 넘어, 설명 가능한 AI(XAI) 기술인 SHAP을 결합하여 **현장 작업자가 즉각적으로 오류 원인을 파악하고 정비에 착수할 수 있는 실질적인 가이드라인(Action Plan)** 을 제공하는 데 집중합니다.

| 이상탐지 결과 (AI) | SHAP 원인 분석 (XAI) | 도메인 액션 플랜 (현장 지시) |
| :--- | :--- | :--- |
| **Test06_NG** (이상 판정) | `M16CV11` 센서 전압 이상 기여도 압도적 1위 | "특정 셀의 과전압/과충전 감지. **해당 모듈 전압 밸런싱 및 센싱 와이어 최우선 점검**" |
| **Test07_NG** (이상 판정) | `M09T01` 센서 온도 가중치 이상 기여 | "특정 모듈의 온도 급하강 감지. **해당 영역 서미스터(Thermistor) 센서 오작동 및 냉각계통 점검**" |

---

## 📂 저장소 구조 (Repository Structure)
프로젝트는 유지보수와 확장성을 극대화하기 위해 역할 기반으로 모듈화되어 관리됩니다.

```text
Battery_Anomaly_Detection/
│
├── .gitignore                      # Git 관리 제외 파일 설정
├── requirements.txt                # 개발 환경 패키지 명세서
│
├── src/                            # 모듈화된 핵심 소스코드
│   ├── config.py                   # 하이퍼파라미터, 센서 칼럼 명세 및 경로 관리
│   ├── dataset.py                  # PyTorch Dataset 정의
│   ├── model.py                    # DistilBert 기반 Transformer Autoencoder 아키텍처
│   ├── preprocess.py               # 결측치 처리, 상수 센서 필터링 및 스케일링 파이프라인
│   ├── train.py                    # 모델 학습 및 Early Stopping 기반 검증 로직
│   └── evaluate.py                 # 테스트셋 추론, F1-Score 산출 및 SHAP 기반 원인 규명
│
├── notebooks/                      # 실험 및 프로토타이핑 기록
│   └── Transfomer4battery.ipynb
│
└── docs/                           # 도메인 지식 및 데이터 세부 가이드
    ├── 데이터구조.txt
    └── Guidebook_전자부품(배터리팩) 품질보증 AI 데이터셋.pdf
    └── 제출_산업경영공학과 인공지능 및 데이터사이언스 경진대회_20181042김경순.pdf
```


## 🛠️ 핵심 방법론 및 아키텍처 (Methodology)
<img width="1189" height="590" alt="f1_score" src="https://github.com/user-attachments/assets/d063fb1b-bf70-4409-bfd7-c601f85803f1" />

### 1. Robust 데이터 전처리 파이프라인
- **결측치 제어**: 산업 데이터 특성상 발생하는 결측치 및 인피니티 에러를 안전하게 0.0으로 대체하는 `SimpleImputer` 적용 및 상태 저장.
- **상수 센서 필터링**: 시계열 전반에 걸쳐 값이 변하지 않는 무의미한 센서(표준편차 $10^{-9}$ 미만)를 동적으로 탐지하여 데이터 차원에서 영구 제외(208개 채널 중 유효 채널 자동 추출).
- **데이터 표준화**: 유효 센서 채널에 대하여 `StandardScaler`를 적용하여 이질적인 물리량(전압, 온도) 간의 스케일을 통일.

### 2. DistilBert 기반 Transformer Autoencoder
- **시퀀스 인코딩**: 208개 전압 및 온도 채널의 다변량 데이터를 임베딩 레이어를 통해 고차원($d_{model}=128$) 공간으로 투사 후, 포지셔널 임베딩을 결합하여 채널 간의 상대적 정렬 상태를 표현.
- **Self-Attention 백본**: Hugging Face의 `DistilBertModel`(4 Layers, 4 Heads, Hidden Dim 256) 아키텍처를 차용하여 센서 간 유기![Uploading f1_score.png…]()
적인 상관관계와 정상 상태의 내부 동기화 패턴을 고도화된 어텐션 메커니즘으로 학습.
- **이상치 점수(Anomaly Score)**: 정상 데이터로만 학습된 오토인코더가 불량 패턴을 입력받을 때 발생하는 **복원 오차(MSE Loss)**의 크기를 기반으로 이상 수준을 정량화.

### 3. Precision-Recall 곡선 기반 탐색
- 통계적 임계값 설정의 한계를 극복하기 위해, 실제 정답 라벨이 존재하는 테스트 파일별로 Precision-Recall 곡선을 그려 **F1-Score를 최대로 만드는 동적 최적 임계값 및 최대 달성 F1-Score**를 수학적으로 역산하여 모델의 진단 한계를 정밀 검증.

### 4. 설명 가능한 AI (SHAP) 기반 불량 원인 규명
- **배경 데이터 축소**: 고차원 모델의 SHAP 연산 오버헤드를 극복하기 위해 정상 훈련 데이터셋을 대표하는 10개의 핵심 군집 센트로이드를 `KMeans`로 추출하여 연산 병목을 최적화.
- **원인 분석**: 최종 복원 오차(Reconstruction Error) 자체를 Target 함수로 지정하고 `KernelExplainer`를 통해 각 센서 채널이 오차 증가에 기여한 섀플리 값(Shapley Value)을 계산. 이를 통해 불량을 유발한 최우선 순위 '범인 센서(Culprit Sensor)'를 특정.
<img width="1002" height="494" alt="test07_NG_dchg_M02CV02" src="https://github.com/user-attachments/assets/36d0495b-0677-4dcb-958f-9f0787e0ff89" />
<img width="1002" height="494" alt="test07_NG_dchg_M02CV01" src="https://github.com/user-attachments/assets/cc718659-dd5d-4d82-bdae-baa42e903937" />


---

## 🚀 시작하기 (How to Run)

### 1. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 디렉토리 설정 및 데이터 준비
구글 드라이브 또는 로컬 환경에 맞게 src/config.py 파일 내의 BASE_PATH 경로를 수정합니다.
```Python
# src/config.py
BASE_PATH = "/content/drive/MyDrive/KAMP_Battery_Project"
```

### 3. 파이프라인 실행
순수 파이썬 모듈 환경에서 전처리, 학습 및 검증, 최종 평가까지 한 줄로 안전하게 실행할 수 있습니다.

모델 학습 및 가중치 저장:
```Bash
python src/train.py
```
학습이 완료되면 지정된 경로에 전처리 객체(.pkl)와 베스트 모델 가중치(.pth)가 자동으로 저장됩니다.

테스트셋 추론 및 SHAP 요약 결과 도출:
```Bash
python src/evaluate.py
```

실행 완료 시, 각 불량 파일에 대한 F1-Score 리포트가 출력되며 최우선 원인 센서들이 요약된 shap_summary.png 시각화 파일이 디렉토리에 생성됩니다.
