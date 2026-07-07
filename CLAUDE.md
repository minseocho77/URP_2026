# CLAUDE.md

이 파일은 이 저장소에서 작업하는 Claude Code에게 주는 안내다.

## 프로젝트 개요
Zhuang et al. (2022), *Computers in Industry* 143, 103747 — 탄소포집 파일럿 플랜트
흡수탑의 6개 지점 CO₂ 농도를 추정하는 하이브리드(메커니즘 + 데이터 기반) 소프트센서
논문의 **재현/구현** 프로젝트다. 원저자 코드는 Jupyter 노트북(Keras2/pandas1, 2022)이며,
이를 2026년 스택(Keras3/pandas3/numpy2)에서 동작하도록 `.py`로 옮기고 디버깅했다.

핵심 모델 3종의 융합:
1. **Mechanistic** — 5단 향류 CSTR 흡수탑 (원본 MATLAB/Simulink → Python 이식 완료)
2. **DAE-LSTM** — DAE로 90→16 차원축소 후 LSTM 시계열 추정
3. **Fused** — 두 모델의 오차공분산(B,R)과 3D-Var(칼만)로 결합 (Eq.17)

## 환경 / 실행
가상환경은 저장소 내 `.venv` (Python 3.12). **항상 이 인터프리터로 실행**한다.
```powershell
.\.venv\Scripts\python.exe <script.py>
```
설치 패키지: numpy pandas matplotlib scikit-learn scipy openpyxl tensorflow tf-keras
(SSAE 베이스라인용 torch는 미설치 — 재현 범위 외, 논문값 인용)

### 주요 스크립트
| 스크립트 | 역할 |
|---|---|
| `reproduce.py` | 전처리 → 차원축소(DAE/POD/PCA) → LSTM 학습 → 융합 → 논문값 대조 |
| `kinetic_model_py/kinetic_model.py` | MATLAB 메커니즘 모델의 Python 이식본. `csv_py/*.csv` 생성 |
| `make_fig11.py` | 논문 Fig.11 재현 그림(`fig11_repro.png`) 생성 |

### reproduce.py 환경변수
| 변수 | 의미 | 기본 |
|---|---|---|
| `REPRO_MODE` | 차원축소 {DAE, POD, PCA} | DAE |
| `REPRO_DIM` | 축소 차원 {16, 32} | 16 |
| `REPRO_TEST` | 시나리오 {1, 2, 3} (Set-1/2/3) | 3 |
| `REPRO_KINETIC_DIR` | 메커니즘 출처. `kinetic_model_py/csv_py`=Python 이식본 | kinetic_model |
| `REPRO_EPOCHS_DAE` / `REPRO_EPOCHS_LSTM` | 학습 epoch | 500 / 500 |

빠른 파이프라인 점검은 epoch를 50/50으로 낮춰 실행한다. CPU 학습이라 500/500은 수 분 소요.
출력이 길고 학습이 오래 걸리므로 **백그라운드 실행 후 결과 표만 확인**하는 방식을 권장.

## 아키텍처 / 논문↔코드 매핑
- 전처리: `avgOutPoint1`(지점1 보정), `columnSeparator`(선형보간), MinMaxScaler(Eq.1)
- 차원축소: `dfAE`(DAE, Table1/Eq.5-8), `dfPOD`(SVD, Eq.3-4), PCA는 feature 선택
- 시퀀스: `getSampleSet` — 이동창 `callback+1=18`, feature=인코딩16 + one-hot6 = 22, 라벨=6지점
- LSTM: `trainSequenceLSTM` — LSTM(100)+Dropout(0.1)×3+Dense(6,sigmoid), Adam lr=1e-5, MSE
  출력 `(n,18,6)` → `np.mean(axis=1)` → 역정규화. 검증셋으로 R=Cov(y_LSTM−y) 추정(Eq.18-19)
- 융합: `run_fusion`(B=Cov(y_mec−y), Gaspari-Cohn `covLoc`/`GCfunc`), `VAR_3D`(Eq.17)
- 메커니즘: `kinetic_model.py` `stage_deriv`(6상태 ODE) + `column_deriv`(5단×6=30상태 연립)

## 중요 주의사항 (gotcha)
- **numpy2/pandas3 패치 2건** (`reproduce.py`에 `[PATCH]` 표시):
  ① pandas3 `.values`는 read-only → in-place 대입 전 `np.array(..., copy=True)` 필요.
  ② 융합엔 LSTM raw `(n,Nseq,6)`가 아니라 시퀀스평균+역정규화 `(n,6)`을 넘겨야 함.
- **Keras2→Keras3 재현성 한계**: seed 고정해도 LSTM 가중치가 재현 안 됨 → DAE-LSTM RMSE가
  논문보다 높은 유일한 원인. 이건 코드 버그가 아님. 메커니즘/전처리/융합은 결정론적으로 재현됨.
- **Fused ≈ Mechanistic 수렴은 정상**: LSTM 노이즈↑ → R↑ → 융합이 메커니즘에 가중(down-weight).
- **MATLAB 컬럼 매핑**: `results_generation.m`의 `input_df(:,k)` == pandas `df.iloc[:,k]`.
- **원본 데이터/CSV는 수정 금지**: `data/withLabel/*.xlsx`, `kinetic_model/*.csv`(원저자 MATLAB 출력),
  `results/*.npy`(원저자 SSAE)는 입력/참조용. 재현 산출물은 `*_repro.*`, `csv_py/`, `fig11_*` 로 분리.

## 최종 재현 결과 (요약)
Mechanistic(Python 이식): Set-1 0.114 / Set-2 0.325 / Set-3 0.189 — 논문(0.117/0.325/0.190) 거의 일치.
상세는 `FINAL_REPORT.md` 참조.
