# Battery-Anomaly-Detection-XAI
> 🏭 Pure PyTorch Transformer Autoencoder와 SHAP을 활용한 배터리팩 이상탐지 및 현장 액션 플랜 도출 파이프라인

![Award](https://img.shields.io/badge/Award-2025_성결대_데이터경진대회_금상(1위)-gold)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)

## 💡 비즈니스 임팩트 : "데이터를 넘어 현장의 Action으로"
본 프로젝트는 단순한 이상탐지를 넘어, XAI(SHAP)를 통해 **현장 작업자가 즉각적으로 실행 가능한 점검 지표(Action Plan)를 도출**하는 데 집중했습니다.

| 이상탐지 결과 (AI) | SHAP 원인 분석 (XAI) | 도메인 액션 플랜 (현장 지시) |
| :--- | :--- | :--- |
| **Test07_NG** (이상 판정) | `M02CV02`, `M02CV01` 센서 전압 이상 기여도 1위 | "인접 배터리 셀 전압 이상. **해당 모듈 센싱 와이어 단선/불량 최우선 점검**" |
| **Test09_NG** (이상 판정) | `M16T02`, `M16T01` 센서 온도 이상 기여도 1위 | "특정 모듈 온도 편차 발생. **배터리 냉각 시스템 및 온도 센서 오작동 점검**" |

<img width="808" height="873" alt="센싱와이어 불량" src="https://github.com/user-attachments/assets/b797bee2-9e23-4f93-a937-48489b7ac6b9" />

---

## 📈 모델 성능 및 신뢰성 증명 (Anti-Data Leakage)
현업의 엄격한 검증 기준을 충족하기 위해 **학습(Train) / 임계값 탐색(Validation) / 최종 평가(Test) 데이터를 파일 단위로 완전히 격리(3-Split)**하여 과적합 및 데이터 누수(Data Leakage)를 원천 차단했습니다.

- **Model:** Native PyTorch Transformer Autoencoder (Attention 메커니즘을 통한 208개 센서 간 상호작용 학습)
- **Performance:** **최종 미지의 Test Set에서 F-Score 0.99** 달성
- **Robustness:** 기존 PCA 기반 차원 축소의 한계(선형성 변환으로 인한 정보 손실)를 극복하고, 시계열 Raw Data의 특성을 온전히 보존.

### 📊 Reconstruction Error (Anomaly Score) 변별력 증명
`src/evaluate.py` 실행 시 정상(OK) 데이터와 불량(NG) 데이터 간의 오차 분포를 분리하여 임계값(Threshold)의 타당성을 시각적으로 자동 증명합니다.

<img width="1000" height="600" alt="학습 그래프" src="https://github.com/user-attachments/assets/6b1bb521-c814-41fc-99c1-11f2fa4294cd" />
*(위 그래프는 예시이며, `src/evaluate.py` 구동 시 `docs/anomaly_distribution.png` 경로에 실시간 생성됩니다.)*

---

## 🧠 왜 Transformer와 SHAP인가? (Model & XAI Architecture)
배터리팩 데이터는 208개의 센서가 시간에 따라 상호작용하는 복잡한 시계열 데이터입니다. 본 파이프라인은 정교한 딥러닝 아키텍처와 임계값 최적화를 통해 실무적인 이상탐지 시스템을 구현했습니다.

<details>
<summary><b>💡 (클릭) 수학적 엄밀성 및 아키텍처 상세 보기</b></summary>
<div markdown="1">

#### 1. Pure PyTorch Transformer Autoencoder
- 거대한 자연어처리(NLP) 라이브러리(`transformers`)에 의존하지 않고, 순수 PyTorch 내장 모듈(`nn.TransformerEncoder`)로 경량화 아키텍처를 구현하여 추론(Inference) 효율성을 극대화했습니다.
- **Self-Attention Mechanism**: 
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  각 센서 데이터가 다른 센서들과 가지는 상관관계를 가중치로 학습하여 정상 상태의 동기화 패턴을 정의합니다.
- **Anomaly Score**: 정상 데이터로만 학습된 Autoencoder는 미지의 불량 패턴 입력 시 복원 오차(MSE)가 급증하며, 이 에러 값을 이상치 점수로 활용합니다.

#### 2. F-Score 기반 동적 임계값(Threshold) 탐색
- 통계적 가정을 따르는 대략적인 기준(3-Sigma 등) 대신, 검증셋(정상 1개 + 불량 1개 파일)의 Precision-Recall 곡선 상에서 **F1-Score를 최대로 만드는 최적의 Threshold 지점을 수학적으로 추적**하여 고정합니다.

#### 3. 불량 원인 규명 및 연산 최적화 (SHAP)
- 모델의 최종 복원 오차를 Target Value로 설정하고, 208개 센서 입력에 대한 섀플리 값(Shapley Value)을 계산합니다.
  $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$
- 대용량 시계열 모델의 SHAP 연산 오버헤드를 해결하기 위해, 무거운 연산 결과를 사전 추출하여 캐싱(`models/shap_values.npy`)하는 구조를 채택했습니다. 이를 통해 실무 시연 시 대기 시간을 0.1초로 단축했습니다.

</div>
</details>

---

## 🚀 Quick Start (재현 방법)
본 프로젝트는 중앙 집중식 설정 파이프라인(`config.yaml`)을 기반으로 완벽하게 모듈화되어 있습니다.

### 1. 데이터 다운로드
데이터 저작권 및 용량 관계로 샘플 데이터만 제공됩니다. 전체 원본 데이터는 아래 링크에서 다운로드 후 `data/sample/` 폴더에 배치해 주세요.
- **데이터 출처:** [KAMP 전자부품(배터리팩) 품질보증 AI 데이터셋](https://www.kamp-ai.kr/aidataDetail?AI_SEARCH=&page=5&DATASET_SEQ=58&DISPLAY_MODE_SEL=CARD&EQUIP_SEL=&GUBUN_SEL=&FILE_TYPE_SEL=&WDATE_SEL=)

### 2. 환경 세팅
```bash
# 가상환경 생성 및 활성화
conda create -n battery_xai python=3.8
conda activate battery_xai

# 의존성 패키지 설치
pip install -r requirements.txt
