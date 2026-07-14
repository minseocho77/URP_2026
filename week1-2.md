# 하이브리드 CO₂ 소프트센서 — 1–2주차 산출물

> 목적: 학습 모델을 붙이기 **전에** 물리 파이프라인을 검증하고(1주차), grey-box 역문제로 물리를 de-risk하면서 PINN loss 인프라를 완성(2주차)한다.
>
> 무결성 노트: 이 문서는 **프로토콜·수식·골격**이다. RHS 조립, loss 가중치 튜닝, 학습 루프의 세부는 TODO로 남겼다 — 그 부분이 프로젝트의 학습 핵심이자 방어 근거이므로 직접 구현할 것.

---

## 산출물 A (1주차) — Forward 시뮬레이션 검증 프로토콜

### A.0 왜 먼저 하나
논문의 기계론적 ODE를 코드로 옮긴 게 "맞다"는 확신 없이 NN을 붙이면, 나중에 오차가 나도 물리 버그인지 학습 문제인지 구분할 수 없다. 이 단계 통과가 뒤 5주의 전제다.

### A.1 구축 순서
1. **상태 정의**: `x = [C_g,1..5, C_l,1..5]` (등온 가정, 온도는 상수 입력으로). 확장 시 `T_l,1..5` 추가.
2. **RHS 조립** — 논문 Eq.10–11의 stage-wise 이산화:
   - 기체: `dC_g,j/dt = [F_g·(C_g,{j-1} − C_g,j)/V_sec − c·R_j] / ε_g`
   - 액체: `dC_l,j/dt = [F_l·(C_l,{j+1} − C_l,j)/V_sec + c·R_j] / ε_l`
   - 경계: bottom(j=5)의 `C_g,in` = 측정 inlet, top(j=1)의 `C_l,in` = lean solvent.
3. **상수 주입** (논문 Table 2): `ε_l=0.035, ε_g=0.951, V_sec, D_i=0.0507, ρ_l=1004, C_pl=3.4145` 등. 반응속도는 Table 4 값(k₁=500.6, α₁=1.0, β₁=1005.5, k₂=4380, α₂=1.0, β₂=1263)을 **하드코딩**해서 논문 파라미터로 재현.
4. **입력 스케줄**: F_g, F_l, inlet CO₂를 데이터에서 읽어 시간의 함수로. inlet은 논문 방식대로 다음 cycle까지 이전 값 유지(piecewise-constant hold).
5. **적분**: 강성 대비 `dopri5` 또는 stiff solver, 상태·시간 nondimensionalize.

### A.2 통과 기준 (Acceptance criteria)

| 검증 항목 | 방법 | 통과 기준 |
|---|---|---|
| 비음수성 | 전 구간 min(C) | 모든 농도 ≥ 0 (수치오차 제외) |
| 프로파일 단조성 | 상단으로 C_g 감소 | 논문 heuristic과 일치 |
| 질량수지 폐합 | in − out − 반응소모 | ≈ 0 (잔차 < 1e-3 규모) |
| **mechanistic RMSE 재현** | Set-1, Set-2에서 논문 프로토콜대로 | Set-1 ≈ 0.117, Set-2 ≈ 0.325 (±20% 이내) |
| inlet staircase 거동 | inlet hold 후 응답 | 논문 Fig.11 상단 거동과 정성적 일치 |

> 핵심 게이트: **mechanistic RMSE 재현**이 안 되면 A.3으로 가서 원인 격리 후 반복. 여기서 멈추고 NN으로 넘어가지 말 것.

### A.3 흔한 실패 모드 → 진단
- 발산/NaN → 강성 문제. 시간 스케일 조정, solver 교체, 상태 정규화.
- RMSE는 크지만 형태는 맞음 → 상수 단위 불일치(몰/부피 환산, 이상기체 적용 여부) 점검.
- 하단만 크게 틀림 → inlet hold 로직 또는 경계조건 버그. 논문도 하단 stage가 RMSE를 지배한다고 명시.
- 프로파일이 뒤집힘 → 기체 상승/액체 하강 방향(j−1 vs j+1) 부호 확인.

### A.4 1주차 말 산출물
- [ ] `mechanistic_model.py` — RHS + odeint forward
- [ ] 검증 노트북: 위 표 5개 항목 통과 스크린샷/수치
- [ ] baseline LSTM 재현 RMSE (비교군 확보)

---

## 산출물 B (2주차) — Grey-box 역문제 + Loss 정의

### B.0 목적
`k, α, β`를 **데이터로 추정**해서 Table 4와 대조. (1) 물리가 미분가능 파이프라인에서 학습 가능한지 검증하고, (2) 3–4주차 hybrid UDE가 그대로 재사용할 loss 인프라를 완성한다.

### B.1 상태·파라미터
- 상태: A.1과 동일.
- 학습 파라미터 θ = {k₁, k₂, α₁, α₂, β₁, β₂}. 양수 보장을 위해 `k = exp(log_k)`로 재매개변수화.
- 활성화에너지 E_A1, E_A2는 논문처럼 **고정**(복잡도 축소).

### B.2 Loss 정의 (수식)

전체 목적함수:

```
L(θ) = L_data + λ_p · L_phys + λ_b · L_bc
```

- 데이터 항 (라벨된 (stage s, time t)에서만; 논문 stage 가중 w_s = [0,20,10,7,1] 차용):
  ```
  L_data = (1/Σmask) · Σ_{s,t} mask_{s,t} · w_s · (ŷ_{s,t} − y_{s,t})²
  ```
- 물리/제약 항 (**라벨 불필요** — collocation 점에서):
  ```
  L_phys = mean( relu(−C)² )                     # 비음수
         + mean( relu(C_g,{j+1} − C_g,j)² )      # 상단으로 단조감소(방향 주의)
  ```
- 경계 항:
  ```
  L_bc = mean( (Ĉ_g,inlet − C_inlet,measured)² )
  ```

> λ_p, λ_b는 고정으로 시작. 학습이 한 항에 지배되면 adaptive weighting(learning-rate annealing / uncertainty weighting)을 직접 구현 — 이 튜닝 로그가 보고서 discussion의 핵심 소재.

### B.3 코드 골격 (torchdiffeq, TODO 포함)

```python
import torch, torch.nn as nn
from torchdiffeq import odeint

class GreyBoxODE(nn.Module):
    def __init__(self, consts, inputs_fn):
        super().__init__()
        # 학습 파라미터 (양수 보장 위해 log 공간)
        self.log_k1 = nn.Parameter(torch.tensor(0.0))
        self.log_k2 = nn.Parameter(torch.tensor(0.0))
        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.a2 = nn.Parameter(torch.tensor(1.0))
        self.b1 = nn.Parameter(torch.tensor(1000.0))
        self.b2 = nn.Parameter(torch.tensor(1200.0))
        self.consts = consts          # Table 2 상수 dict
        self.inputs_fn = inputs_fn     # t -> (F_g, F_l, C_inlet, T)

    def rates(self, Cg, Cl, T):
        k1, k2 = self.log_k1.exp(), self.log_k2.exp()
        # TODO: 수정 Arrhenius로 R1, R2 (논문 Eq.12-13), 순 R = R1 - R2
        # R1 = k1 * exp(-EA1 / (8.314*(self.a1*T + self.b1)))
        raise NotImplementedError

    def forward(self, t, x):
        Cg, Cl = x[..., :5], x[..., 5:10]
        Fg, Fl, C_in, T = self.inputs_fn(t)   # TODO: 시간보간/hold 구현
        R = self.rates(Cg, Cl, T)
        # TODO: A.1의 balance로 dCg/dt, dCl/dt 조립 (경계항 포함)
        # 상승기체 j-1, 하강액체 j+1 방향 주의
        raise NotImplementedError
        return dxdt

def loss_fn(pred, obs, mask, w_s, lam_p, lam_b, C_in_meas):
    # TODO: B.2 수식 그대로 구현
    raise NotImplementedError

# 학습 루프 (plumbing)
func = GreyBoxODE(consts, inputs_fn)
opt = torch.optim.Adam(func.parameters(), lr=1e-2)
for epoch in range(N):
    opt.zero_grad()
    pred = odeint(func, x0, t_eval, method='dopri5')
    loss = loss_fn(pred, obs, mask, w_s, lam_p, lam_b, C_in_meas)
    loss.backward(); opt.step()
```

> 남긴 3개 `NotImplementedError`가 곧 여러분이 화학·수치를 이해했다는 증거다. plumbing(odeint 호출·루프)은 채워뒀다.

### B.4 파라미터 추정 절차
1. 초기값: log_k는 0 근처, α≈1, β를 Table 4 규모로 (물리적 초기점).
2. Set-3에서 추정 (논문도 Set-3로 patternsearch 피팅).
3. 수렴 후 Set-1/2에서 검증 RMSE.
4. 발산 시 lr 축소·gradient clipping, 또는 β를 먼저 고정하고 k만 추정하는 2단계 워밍업.

### B.5 검증 — Table 4 대조

| 파라미터 | 논문 값 | 내 추정값 | 상대오차 |
|---|---|---|---|
| k₁ | 500.6 | — | — |
| α₁ | 1.0 | — | — |
| β₁ (K) | 1005.5 | — | — |
| k₂ | 4380.0 | — | — |
| α₂ | 1.0 | — | — |
| β₂ (K) | 1263.0 | — | — |

> 추정값이 물리적으로 타당한 범위에 들어오면 "블랙박스가 아니라 반응속도를 학습했다"는 강한 근거. 크게 벗어나면 identifiability 문제(데이터가 파라미터를 구속 못 함)일 수 있음 — 이것도 훌륭한 discussion 소재.

### B.6 2주차 말 산출물
- [ ] `greybox_ode.py` — 미분가능 forward + 파라미터 추정
- [ ] Table 4 대조표(값 채움) + 수렴 곡선
- [ ] `loss_fn` 완성 (3–4주차 hybrid에서 재사용)
- [ ] λ 스윕 예비 실험 1회

---

## 두 주 끝의 상태 점검
이 두 산출물이 끝나면: (1) 물리가 검증됐고, (2) 미분가능 파이프라인·loss가 돌아가며, (3) 남은 건 `rates()`의 Arrhenius를 신경망 `Rθ`로 교체하는 일뿐 — 3–4주차 hybrid UDE로 매끄럽게 이어진다.