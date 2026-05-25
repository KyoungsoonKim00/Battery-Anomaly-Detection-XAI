# Battery-Anomaly-Detection-XAI
> 🏭 Transformer와 SHAP을 활용한 배터리팩 이상탐지 및 현장 액션 플랜 도출 파이프라인

![Award](https://img.shields.io/badge/Award-2025_성결대_데이터경진대회_금상(1위)-gold)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)

## 💡 비즈니스 임팩트 : "데이터를 넘어 현장의 Action으로"
본 프로젝트는 단순한 이상탐지를 넘어, XAI(SHAP)를 통해 **현장 작업자가 즉각적으로 실행 가능한 점검 지표를 도출**하는 데 집중했습니다.

| 이상탐지 결과 (AI) | SHAP 원인 분석 (XAI) | 도메인 액션 플랜 (현장 지시) |
| :--- | :--- | :--- |
| **Test07_NG** (이상) | `M02CV02`, `M02CV01` 센서 전압 이상 기여도 1위 | "인접 배터리 셀 전압 이상. **해당 모듈 센싱 와이어 단선/불량 최우선 점검**" |
| **Test09_NG** (이상) | `M16T02`, `M16T01` 센서 온도 이상 기여도 1위 | "특정 모듈 온도 편차 발생. **배터리 냉각 시스템 및 온도 센서 오작동 점검**" |

 <img width="808" height="873" alt="센싱와이어 불량" src="https://github.com/user-attachments/assets/b797bee2-9e23-4f93-a937-48489b7ac6b9" />

---

## 📈 모델 성능 및 신뢰성 증명
단순히 F-score 0.99라는 수치에 그치지 않고, 정상(OK) 데이터와 불량(NG) 데이터 간의 Anomaly Score 분포를 명확히 분리하여 모델의 변별력을 수학적으로 증명했습니다.

<img width="1000" height="600" alt="학습 그래프" src="https://github.com/user-attachments/assets/6b1bb521-c814-41fc-99c1-11f2fa4294cd" />


- **Model:** Transformer Autoencoder (Attention 메커니즘을 통한 센서 간 상호작용 학습)
- **Performance:** **F-Score 0.99** 달성
- **Robustness:** PCA 기반 차원 축소의 한계(정보 손실)를 극복하고, 208개 센서 Raw Data의 시계열적 특성을 온전히 보존.

---

## 🚀 Quick Start (재현 방법)
본 프로젝트는 누구나 쉽게 결과를 재현할 수 있도록 모듈화되어 있습니다.

### 1. 데이터 다운로드
데이터 용량 문제로 샘플 데이터(`data/sample/`)만 제공됩니다. 전체 원본 데이터는 아래 링크에서 다운로드 후 `data/raw/` 폴더에 배치해 주세요.
- **데이터 출처:** [KAMP 전자부품(배터리팩) 품질보증 AI 데이터셋](https://www.kamp-ai.kr/aidataDetail?AI_SEARCH=&page=5&DATASET_SEQ=58&DISPLAY_MODE_SEL=CARD&EQUIP_SEL=&GUBUN_SEL=&FILE_TYPE_SEL=&WDATE_SEL=)

### 2. 환경 세팅
```bash
# 가상환경 생성 및 활성화 (권장)
conda create -n battery_xai python=3.8
conda activate battery_xai

# 의존성 패키지 설치
pip install -r requirements.txt
