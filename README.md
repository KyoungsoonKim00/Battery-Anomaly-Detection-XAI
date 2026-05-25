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
## 🧠 왜 Transformer와 SHAP인가? (Model & XAI Architecture)
기존의 배터리 이상탐지는 주로 PCA(주성분 분석) 등을 통해 다차원 센서 데이터를 축소하여 분석했습니다. 하지만 배터리팩 데이터는 208개의 센서가 시간에 따라 상호작용하는 복잡한 시계열 데이터이므로, 단순 차원 축소는 중요한 비선형적 패턴과 시간적 흐름(Information Loss)을 놓치게 됩니다.

본 프로젝트는 **Transformer의 Attention 메커니즘**을 도입하여 208개 센서 간의 유의미한 상관관계를 원본 그대로 학습하고, **SHAP**을 통해 블랙박스 모델의 판단 근거(어떤 센서가 이상을 유발했는지)를 역추적하여 현장에 제공합니다.

<details>
<summary><b>💡 (클릭) 수학적 엄밀성 및 아키텍처 상세 보기</b></summary>
<div markdown="1">

#### 1. Transformer Autoencoder
- **Self-Attention Mechanism**: 
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  각 센서 데이터(Query)가 다른 모든 센서(Key)와 얼마나 연관되어 있는지 가중치를 계산합니다. 이를 통해 정상 상태에서의 센서 간 동기화 패턴을 학습하고, 특정 센서의 비정상적인 거동을 민감하게 포착합니다.
- **Anomaly Detection**: 정상(OK) 데이터로만 학습된 Autoencoder는 불량(NG) 데이터 입력 시 복원 오차(Reconstruction Error)가 급증합니다. 이 MSE(Mean Squared Error) 기반의 오차를 최종 Anomaly Score로 활용합니다.

#### 2. 불량 원인 규명 (SHAP)
- 모델의 최종 복원 오차를 Target Value로 설정하고, 208개 입력 피처(센서)에 대한 섀플리 값(Shapley Value)을 계산합니다.
  $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$
- 이 수식을 통해 특정 센서가 최종 이상 점수 도출에 기여한 '한계 기여도'를 정량화하여, "M16T02 센서가 전체 이상 점수 증가에 45% 기여했다"와 같은 명확한 해석을 가능하게 합니다.

</div>
</details>


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
