# Loss 가중치 튜닝 트래커 (산출물 D / wekk3-4.md)

> `L = L_data + λ_p·L_phys + λ_b·L_bc` 에서 각 항의 **gradient 크기 불균형**이
> PINN 계열 실패의 1순위 원인. 항별 loss + grad-norm 을 로깅하며 튜닝한다.

## 방법 스펙트럼 (D.2 — 아래로 갈수록 정교)
1. **고정 그리드**: λ ∈ {0.01, 0.1, 1, 10} 스윕 (항상 baseline)
2. **초기 크기 정규화**: 각 항을 첫 epoch 크기로 나눠 동일 스케일 출발
3. **LR annealing (Wang 2021)**: 주기적 grad norm 측정 → λ 자동 균형
4. **Uncertainty weighting (Kendall)**: 항별 학습가능 log-분산
5. **커리큘럼**: 초반 물리 강조 → 후반 데이터 강조 (또는 반대)

## 프로토콜 (D.3)
- 매 epoch **항별 loss + 항별 grad norm** 로깅 (`hybrid_ude.train` 의 로깅 훅 확장).
- 진단: 한 항 grad norm 이 다른 항의 **10배↑** 이면 불균형 → 정규화/adaptive.
- 고정 그리드로 최적 λ 찾고, **강건성**(같은 λ 가 시드·Set 넘어 유지되나)도 보고.
- "고정-최적 vs adaptive" 비교 = 논문 그림 1장.

## 튜닝 트래커 (값 채우기)

| run | λ_p | λ_b | weighting | seed | Val RMSE | Test RMSE | grad균형(≤10배) | 메모 |
|---|---|---|---|---|---|---|---|---|
| d01 | 1 | 1 | fixed | 0 | — | — | — | baseline |
| d02 | 0.1 | 1 | fixed | 0 | — | — | — | |
| d03 | 10 | 1 | fixed | 0 | — | — | — | |
| d04 | 1 | 0.1 | fixed | 0 | — | — | — | |
| d05 | 1 | 10 | fixed | 0 | — | — | — | |
| d06 | — | — | init-norm | 0 | — | — | — | 초기크기 정규화 |
| d07 | — | — | LR-anneal | 0 | — | — | — | Wang 2021 |
| d08 | — | — | uncertainty | 0 | — | — | — | Kendall |

## 통과 기준 (D.5)
- [ ] grad norm 불균형 해소(≤10배)된 설정 확보
- [ ] adaptive 가 고정-최적 대비 동등 이상(또는 튜닝 노력 절감)임을 근거와 함께 서술

## 구현 메모 (TODO — 직접 구현 대상)
- `adaptive weighting 핵심`은 학습 핵심이므로 직접 구현 (wekk3-4.md 무결성 노트).
- `hybrid_ude.train` 의 `[TODO] 항별 grad-norm 로깅` 부분을 채워 위 표를 채운다.
- 항별 grad norm: 각 항에 대해 `torch.autograd.grad(term, params, retain_graph=True)` 로
  norm 측정 후, LR-anneal/uncertainty 규칙으로 λ 갱신.
