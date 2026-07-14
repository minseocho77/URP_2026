# -*- coding: utf-8 -*-
"""
==============================================================================
 nn_closure.py  --  [Week-3 / 산출물 C] 신경망 닫힘항 Rθ
------------------------------------------------------------------------------
 2주차 grey-box 의 rates()(수정 Arrhenius) 를 이 NNClosure 로 교체하는 것이
 하이브리드 UDE 의 요점. balance/경계/적분/loss 는 전부 2주차 것 그대로 재사용.

 설계 결정 (wekk3-4.md C.1):
   입력  : 국소 상태 (C_g,j, C_l,j, T_j)  — stage 공유 가중치(동질성 유지)
   출력  : stage 별 순 반응속도 R_j (스칼라)
   크기  : 2 hidden × 16-32, tanh  (데이터 적음 → 작게)
   제약  : 부호/유계성 (TODO — 직접 설계)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import mechanistic_model as M     # 상태 정규화 스케일(MEA_TOT) 재사용


class NNClosure(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),      # [R1, R2] 분리 출력 -> 가역 구조 유지 (C.1)
        )

    def forward(self, Cg, Cl, T):
        """Cg,Cl: (n,) / T: 스칼라 또는 (n,) -> 순 반응 R: (n,)."""
        Texp = T.expand_as(Cg) if torch.is_tensor(T) and T.dim() == 0 else \
               (T if torch.is_tensor(T) else torch.full_like(Cg, float(T)))
        # [C.1 정규화] 상태 스케일 통일 (Cl~O(1e3), T~300 이 raw 로 지배하지 않도록)
        feat = torch.stack([Cg, Cl / M.MEA_TOT, (Texp - 300.0) / 50.0], dim=-1)  # (n,3)
        out = self.net(feat)                       # (n, 2)
        # [C.1 물리 제약] R1,R2 >= 0 (softplus): 흡수/탈착 각각 비음수, 순 R 은 부호 자유.
        R1 = F.softplus(out[..., 0])               # 흡수(정방향)
        R2 = F.softplus(out[..., 1])               # 탈착(역방향)
        return R1 - R2                             # 순 반응속도 (n,)


# --- warm-start 유틸 (wekk3-4.md C.2) : grey-box distill ---------------------
def distill_from_greybox(closure, greybox_rates_fn, n_samples=4000,
                         ranges=None, epochs=500, lr=1e-3, seed=0):
    """
    2주차 피팅된 Arrhenius 를 Rθ 에 사전학습(MSE)해 warm-start (C.2).
      1) 상태공간 (Cg,Cl,T) 를 균일 샘플 (ranges)
      2) target = greybox_rates_fn(Cg,Cl,T)   # 2주차(또는 Table4) 순 반응속도
      3) closure 출력과 MSE 로 epochs 만큼 사전학습
    반환: 학습된 closure (물리적으로 타당한 초기점).
    """
    torch.manual_seed(seed)
    if ranges is None:
        # 상태 물리 범위 (Cg: 기상 CO2 mol/m3, Cl: 결합CO2 0~MEA_TOT, T: 운전온도)
        ranges = dict(Cg=(0.0, 2.5), Cl=(0.0, M.MEA_TOT), T=(295.0, 325.0))
    Cg = torch.empty(n_samples).uniform_(*ranges['Cg'])
    Cl = torch.empty(n_samples).uniform_(*ranges['Cl'])
    T = torch.empty(n_samples).uniform_(*ranges['T'])
    with torch.no_grad():
        target = greybox_rates_fn(Cg, Cl, T)       # (n_samples,) 순 반응속도
    opt = torch.optim.Adam(closure.parameters(), lr=lr)
    for e in range(epochs):
        opt.zero_grad()
        pred = closure(Cg, Cl, T)
        loss = ((pred - target) ** 2).mean()
        loss.backward(); opt.step()
        if e % max(1, epochs // 5) == 0:
            print(f"    [distill] e={e:4d} mse={float(loss):.4e}")
    return closure
