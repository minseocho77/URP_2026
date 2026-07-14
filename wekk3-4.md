# 하이브리드 CO₂ 소프트센서 — 3–4주차 산출물

> 목적: 2주차 grey-box의 Arrhenius `rates()`를 **신경망 닫힘항 Rθ**로 교체하고(3주차), loss 가중치를 튜닝하며, 결측 라벨 실험으로 **"물리가 라벨 부족을 메운다"는 핵심 주장**을 정량 입증한다(4주차).
>
> 재사용 원칙: 보존법칙(balance) 조립·odeint 루프·`loss_fn` 골격은 2주차 산출물(A/B) 것을 **그대로** 쓴다. 바뀌는 건 반응항 계산 한 곳뿐 — 이게 하이브리드 구조의 요점이다.
>
> 무결성 노트: 물리 제약 설계, 구조적 마스킹 로직, adaptive weighting 핵심은 TODO로 남긴다.

---

## 산출물 C (3주차) — Hybrid UDE: NN 닫힘항 Rθ

### C.0 무엇만 바뀌나
2주차 `GreyBoxODE`에서 `self.rates()`가 수정 Arrhenius였다. 하이브리드에서는 이 한 메서드만 `NNClosure`로 교체한다. balance(A.1), 경계조건, 적분, loss 전부 동일 → 버그 표면적이 작다.

### C.1 Rθ 설계 결정

| 항목 | 권장 | 근거 |
|---|---|---|
| 입력 | 국소 상태 `(C_g,j, C_l,j, T_j)` | 반응속도는 국소 물리량 함수. stage 공유 가중치로 기계론 모델의 stage 동질성 유지 |
| 출력 | stage별 순 반응속도 `R_j` (스칼라) | balance에 그대로 대입. 원하면 R₁·R₂ 분리 출력해 가역 구조 유지 |
| 크기 | 2 hidden × 16–32, tanh/softplus | 데이터가 적고 물리가 구조를 지탱 → 작게 |
| 물리 제약 | 부호/유계성 (TODO) | 흡수는 CO₂ 있을 때 정방향, 평형 근처서 역방향. 아키텍처 또는 soft penalty |
| 정규화 | 상태 min-max (논문과 일관) | 스케일 불일치 방지 |

### C.2 Warm-start (강력 권장)
바로 joint 학습에 들어가면 강성·불균형으로 불안정하기 쉽다. **2주차 grey-box를 NN에 distill**한 뒤 시작하라:
1. 상태 공간에서 샘플링(LHS 등).
2. 각 샘플에서 target = 2주차 피팅된 Arrhenius 순 반응속도.
3. Rθ를 MSE로 사전학습해 Arrhenius를 근사.
4. 그 다음 소프트센서 loss로 fine-tune.

이러면 학습이 물리적으로 타당한 초기점에서 출발하고, "grey-box→hybrid 매끄러운 핸드오프"라는 서술도 확보된다.

### C.3 코드 골격 (B.3 plumbing 재사용)

```python
import torch, torch.nn as nn
from torchdiffeq import odeint

class NNClosure(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),   # 순 반응속도 R_j
        )
    def forward(self, Cg, Cl, T):
        # Cg, Cl: (n_stages,) / T: 스칼라 또는 (n_stages,)
        feat = torch.stack([Cg, Cl, T.expand_as(Cg)], dim=-1)  # (n_stages, 3)
        R = self.net(feat).squeeze(-1)
        # TODO: 물리 제약(부호/유계성) 적용 여부 결정 후 구현
        return R

class HybridUDE(nn.Module):
    def __init__(self, consts, inputs_fn, closure):
        super().__init__()
        self.consts, self.inputs_fn, self.closure = consts, inputs_fn, closure
    def forward(self, t, x):
        Cg, Cl = x[..., :5], x[..., 5:10]
        Fg, Fl, C_in, T = self.inputs_fn(t)
        R = self.closure(Cg, Cl, T)
        # TODO: A.1/B.3의 balance를 '그대로' 재사용 (기체 j-1, 액체 j+1)
        raise NotImplementedError
        return dxdt

# --- warm-start: grey-box distill (권장) ---
# TODO: 상태 샘플 -> Arrhenius target -> Rθ MSE 사전학습

# --- joint fine-tune (B의 loss_fn 재사용) ---
func = HybridUDE(consts, inputs_fn, NNClosure())
opt = torch.optim.Adam(func.parameters(), lr=1e-3)
for epoch in range(N):
    opt.zero_grad()
    pred = odeint(func, x0, t_eval, method='dopri5')
    loss = loss_fn(pred, obs, mask, w_s, lam_p, lam_b, C_in_meas)
    loss.backward(); opt.step()
```

### C.4 통과 기준
- 완전 라벨에서 hybrid RMSE ≤ grey-box, 그리고 baseline LSTM(0.201)·fused(0.123) 대비 보고.
- warm-start 유무 학습곡선 비교(안정성 근거).
- 추정된 Rθ가 상태에 대해 물리적으로 타당한 경향(농도↑ → 흡수율↑) 시각화.

---

## 산출물 D (3–4주차) — Loss 가중치 튜닝

### D.1 문제의 본질
`L = L_data + λ_p·L_phys + λ_b·L_bc` 에서 각 항의 **gradient 크기**가 크게 다르다(데이터 잔차는 작고 penalty는 튐). 한 항이 지배하면 나머지가 무시된다. 이게 PINN 계열 실패의 1순위 원인.

### D.2 방법 스펙트럼 (아래로 갈수록 정교)
1. **고정 그리드**: λ를 로그 격자 `{0.01, 0.1, 1, 10}`로 스윕. 항상 baseline으로 보고.
2. **초기 크기 정규화**: 각 항을 첫 epoch 크기로 나눠 동일 스케일서 출발.
3. **Learning-rate annealing (Wang 2021)**: 주기적으로 각 항 gradient norm을 재어 λ를 자동 균형.
4. **Uncertainty weighting (Kendall)**: 항별 학습가능 log-분산으로 가중.
5. **커리큘럼**: 초반 물리 강조(타당 다양체에 구속) → 후반 데이터 강조로 anneal. 또는 반대. 실험 가치 있음.

### D.3 프로토콜
- 매 epoch **각 항 loss + 각 항 gradient norm**을 로깅.
- 진단: 한 항의 grad norm이 다른 항의 10배↑면 불균형 → 정규화/adaptive 적용.
- 고정 그리드로 최적 λ 찾되, **강건성**도 보고(같은 λ가 시드·Set 넘어 유지되나).
- "고정-최적 vs adaptive" 비교 = 논문 그림 1장.

### D.4 튜닝 트래커 템플릿

| run | λ_p | λ_b | weighting | seed | Val RMSE | Test RMSE | grad균형 여부 | 메모 |
|---|---|---|---|---|---|---|---|---|
| d01 | 1 | 1 | fixed | 0 | — | — | — | baseline |
| d02 | 0.1 | 1 | fixed | 0 | — | — | — | |
| d03 | — | — | LR-anneal | 0 | — | — | — | |

### D.5 통과 기준
- grad norm 불균형이 해소(≤10배)된 설정 확보.
- adaptive가 고정-최적 대비 동등 이상(또는 튜닝 노력 절감)임을 근거와 함께 서술.

---

## 산출물 E (4주차) — 결측 라벨 실험 (핵심 주장 입증)

### E.1 왜 헤드라인인가
프로젝트의 novelty는 "물리가 결측 라벨을 메운다"이다. 라벨을 점점 지워도 hybrid는 완만히 degrade하고 순수 데이터 모델(LSTM/SSAE)은 급락함을 **곡선 하나로** 보이는 게 목표.

### E.2 메커니즘 (왜 되나)
라벨을 지우면 `L_data`에서만 빠진다. `L_phys`는 **라벨이 필요 없는 collocation 점**에서 계속 계산되므로 hybrid는 학습 신호가 남는다. 데이터 모델은 그 지점서 신호가 0 → 급락. 이 차이가 곧 증명.

### E.3 두 가지 라벨 제거 전략 (둘 다 실행)
- **랜덤(MCAR)**: 라벨을 균일 확률로 제거. 표준 ablation, 쉬움.
- **구조적(권장·정직)**: 회전 분석기 실제 패턴처럼 **특정 위치/시간블록 통째로** 제거. 실제 배포 상황에 가깝고 더 어려움 → 더 설득력.

### E.4 지표 & 핵심 그림
- 라벨 비율 `{100, 75, 50, 35, 20}%` × 모델 `{hybrid, grey-box, LSTM, SSAE}` → **RMSE vs 라벨비율 곡선**.
- degradation 기울기(라벨 줄 때 RMSE 상승률): hybrid가 가장 평탄해야 함.
- hybrid가 baseline을 역전하는 **교차점**을 명시 → 헤드라인 결과.
- Set-1/2/3 + OOD, **시드 3개↑ 평균±표준편차** (논문화 대비).

### E.5 코드 골격

```python
def make_label_mask(full_mask, keep_fraction, strategy='random', seed=0):
    g = torch.Generator().manual_seed(seed)
    if strategy == 'random':
        keep = (torch.rand(full_mask.shape, generator=g) < keep_fraction)
        return full_mask & keep
    elif strategy == 'structured':
        # TODO: 위치/시간블록 단위로 제거 (회전 분석기 패턴 모사)
        raise NotImplementedError

# 라벨 제거는 L_data에만 영향; L_phys는 collocation 그대로 유지
mask = make_label_mask(full_mask, kf, strategy, seed)
loss = loss_fn(pred, obs, mask, w_s, lam_p, lam_b, C_in_meas)
```

> 주의: 마스크는 **데이터 항에만** 적용. 물리·경계 항에는 절대 마스크를 걸지 말 것 — 그게 이 실험이 성립하는 이유다.

### E.6 통과 기준
- 20% 라벨에서 hybrid가 LSTM/SSAE보다 **명확히** 낮은 RMSE.
- hybrid의 라벨비율-RMSE 곡선이 가장 평탄(기울기 최소).
- 완전 라벨에서 hybrid ≈ 또는 < fused 0.123.

---

## 4주차 말 산출물 / 5주차 핸드오프
- [ ] `nn_closure.py`, `hybrid_ude.py` (warm-start 포함)
- [ ] loss 튜닝 트래커(값 채움) + "고정 vs adaptive" 그림
- [ ] 결측 라벨 곡선(랜덤·구조적, 시드 3개↑) + 교차점 표기
- [ ] 완전 라벨 hybrid RMSE vs {LSTM, grey-box, mechanistic, fused}

이 3개가 끝나면 5주차 전체 평가·ablation·OOD로 바로 넘어간다. 결측 라벨 곡선이 그때 논문의 핵심 그림이 된다.