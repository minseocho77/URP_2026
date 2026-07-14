# -*- coding: utf-8 -*-
r"""
==============================================================================
 greybox_ode.py  --  [Week-2 / 산출물 B] Grey-box 역문제 + Loss 인프라
------------------------------------------------------------------------------
 목적: 반응 파라미터 θ={k1,k2,α1,α2,β1,β2} 를 데이터로 추정해 논문 Table 4 와
       대조한다. 3-4주차 hybrid UDE 가 그대로 재사용할 loss 인프라를 완성한다.

 ┌───────────────────────────────────────────────────────────────────┐
 │  ★ 무결성 노트 (week1-2.md B.3)                                     │
 │  rates() / forward() RHS 조립 / loss_fn() — 3개는 학습 핵심이자     │
 │  방어 근거이므로 **직접 구현**한다. 여기서는 plumbing(파라미터      │
 │  선언·상수·데이터로딩·odeint 호출·학습 루프·Table4 대조)만 채우고,  │
 │  물리·loss 코어는 NotImplementedError + 가이드로 남긴다.            │
 └───────────────────────────────────────────────────────────────────┘

 선행 설치: pip install torch torchdiffeq
   ※ Windows 긴 경로(WinError 206) 실패 시 관리자 PowerShell:
     New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
       -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
"""
import os
import glob
import numpy as np

import torch
import torch.nn as nn
from torchdiffeq import odeint

import mechanistic_model as M   # 1주차 상수/데이터로딩 재사용

torch.set_default_dtype(torch.float64)
N_STAGE = M.N_STAGE

# 논문 stage 가중 (week1-2.md B.2) — 데이터 항 w_s
W_S = torch.tensor([0.0, 20.0, 10.0, 7.0, 1.0])


# ==========================================================================
# Grey-box ODE (week1-2.md B.3) — ★ 물리 코어는 직접 구현 ★
# ==========================================================================
class GreyBoxODE(nn.Module):
    def __init__(self, consts, inputs_fn):
        super().__init__()
        # 학습 파라미터 (양수 보장 위해 log 공간; B.1)
        self.log_k1 = nn.Parameter(torch.tensor(0.0))
        self.log_k2 = nn.Parameter(torch.tensor(0.0))
        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.a2 = nn.Parameter(torch.tensor(1.0))
        self.b1 = nn.Parameter(torch.tensor(1000.0))
        self.b2 = nn.Parameter(torch.tensor(1200.0))
        self.consts = consts          # Table 2 상수 dict
        self.inputs_fn = inputs_fn     # t -> (F_g, F_l, C_inlet, T)

    def rates(self, Cg, Cl, T):
        """
        수정 Arrhenius 순 반응속도 R=R1-R2 (논문 Eq.12-13). torch 미분가능.
        mechanistic_model.reaction_rates 와 동일 물리 — 단 k/α/β 는 '학습 파라미터'.
        """
        EA1, EA2, Rg = self.consts['EA1'], self.consts['EA2'], self.consts['RGAS']
        # 전지수인자는 log 공간 파라미터로(양수 보장, B.1)
        k1 = self.log_k1.exp() * torch.exp(-EA1 / (Rg * (self.a1 * T + self.b1)))
        k2 = self.log_k2.exp() * torch.exp(-EA2 / (Rg * (self.a2 * T + self.b2)))
        free_MEA = M.MEA_TOT - Cl               # 자유 MEA = 총 - 결합CO2
        R1 = k1 * Cg * free_MEA                  # 흡수
        R2 = k2 * Cl                             # 탈착
        return R1 - R2                           # (N_STAGE,)

    def forward(self, t, x):
        Cg, Cl = x[..., :N_STAGE], x[..., N_STAGE:2 * N_STAGE]
        Fg, Fl, C_in, T = self.inputs_fn(t)
        R = self.rates(Cg, Cl, T)
        eps_l, eps_g, V = self.consts['EPS_L'], self.consts['EPS_G'], self.consts['V_SEC']
        # mechanistic_model.rhs 의 벡터화 버전 (동일 물리·방향).
        #   기체 유입: 아래 단(i+1), 최하단=측정 inlet C_in
        Cg_in = torch.cat([Cg[1:], C_in.reshape(1)])
        #   액체 유입: 위 단(i-1), 최상단=lean(결합CO2=0)
        Cl_in = torch.cat([torch.zeros(1, dtype=Cl.dtype), Cl[:-1]])
        dCg = (Fg * (Cg_in - Cg) / V - eps_l * R) / eps_g       # Eq.10
        dCl = Fl * (Cl_in - Cl) / (V * eps_l) + R               # Eq.11
        return torch.cat([dCg, dCl])


def loss_fn(pred_Cg, obs, mask, w_s, lam_p, lam_b, C_in_meas, Cg_inlet):
    """
    [TODO — 직접 구현] week1-2.md B.2 수식 그대로.
      L_data = (1/Σmask) Σ mask*w_s*(ŷ-y)^2           # 라벨된 (stage,time)만
      L_phys = mean(relu(-C)^2) + mean(relu(C_g,{j+1}-C_g,j)^2)   # 비음수+단조
      L_bc   = mean((Ĉ_g,inlet - C_inlet,measured)^2)
    ※ 전부 torch 연산(relu=torch.relu). 3-4주차 hybrid·loss_tuning 이 재사용한다.

    ★ 반환 계약: dict(total, data, phys, bc)  (마스크는 L_data 에만!)
    """
    # --- L_data: 라벨된 (stage,time)만, stage 가중 w_s (B.2) ---
    w = w_s.reshape(1, -1)                                  # (1, N_STAGE)
    m = mask.to(pred_Cg.dtype)
    se = (pred_Cg - obs) ** 2 * w * m
    L_data = se.sum() / m.sum().clamp(min=1.0)
    # --- L_phys: 라벨 불필요(collocation). 비음수 + 상단으로 단조감소 ---
    #   비음수: C_g >= 0
    nonneg = torch.relu(-pred_Cg) ** 2
    #   단조: index 0=top(최저) .. 4=bottom(최고). 위단이 아래단보다 크면(비물리) 벌점.
    mono = torch.relu(pred_Cg[:, :-1] - pred_Cg[:, 1:]) ** 2
    L_phys = nonneg.mean() + mono.mean()
    # --- L_bc: 최하단 기체 ≈ 측정 inlet ---
    L_bc = ((Cg_inlet - C_in_meas) ** 2).mean()
    total = L_data + lam_p * L_phys + lam_b * L_bc
    return dict(total=total, data=L_data, phys=L_phys, bc=L_bc)


# ==========================================================================
# 데이터/입력 plumbing (제공)
# ==========================================================================
def make_inputs_fn(prep):
    """load_inputs 결과 -> torch inputs_fn(t) (ZOH hold)."""
    grid = M.INI_TIME + M.DT_IN * np.arange(prep['n'])
    Fg = torch.as_tensor(prep['Fg']); Fl = torch.as_tensor(prep['Fl'])
    Cin = torch.as_tensor(prep['C_g_in']); T = torch.as_tensor(prep['T'])
    g = torch.as_tensor(grid)

    def inputs_fn(t):
        tt = t.detach() if torch.is_tensor(t) else torch.as_tensor(float(t))
        k = int(torch.searchsorted(g, tt.reshape(()), right=True).item()) - 1
        k = max(0, min(k, prep['n'] - 1))
        return Fg[k], Fl[k], Cin[k], T[k]

    return inputs_fn, torch.as_tensor(grid)


def load_targets(prep):
    """
    관측 obs/mask/inlet 구성 (pred_Cg 와 동일 단위 = mol/m^3).
      obs[t,s]  = (측정 CO2 분율)/Vm[t]   (label[t]==s 인 곳만)
      mask[t,s] = True  (측정된 (t,stage))
      C_in_meas = 측정 inlet (mol/m^3, 각 시점) — bottom 경계 bc 용
    """
    T = prep['n']
    Vm = np.asarray(prep['Vm']); co2 = np.asarray(prep['co2_meas'])
    label = np.asarray(prep['label']).astype(int)
    obs = np.zeros((T, N_STAGE)); mask = np.zeros((T, N_STAGE), dtype=bool)
    for t in range(T):
        s = label[t]
        if 1 <= s <= N_STAGE:
            obs[t, s - 1] = co2[t] / Vm[t]      # 분율 -> mol/m^3
            mask[t, s - 1] = True
    return (torch.as_tensor(obs), torch.as_tensor(mask),
            torch.as_tensor(np.asarray(prep['C_g_in'])))


# ==========================================================================
# 학습 루프 (plumbing 제공) — B.4 절차
# ==========================================================================
def estimate(prep, epochs=300, lr=1e-2, lam_p=1.0, lam_b=1.0):
    inputs_fn, t_grid = make_inputs_fn(prep)
    consts = dict(EPS_L=M.EPS_L, EPS_G=M.EPS_G, V_SEC=M.V_SEC,
                  EA1=M.EA1, EA2=M.EA2, RGAS=M.RGAS)
    func = GreyBoxODE(consts, inputs_fn)
    opt = torch.optim.Adam(func.parameters(), lr=lr)

    x0 = torch.zeros(2 * N_STAGE)
    obs, mask, C_in_meas = load_targets(prep)   # TODO 완성 필요

    for epoch in range(epochs):
        opt.zero_grad()
        pred = odeint(func, x0, t_grid, method='dopri5')     # (T, 10)
        Cg_inlet = pred[:, N_STAGE - 1]
        terms = loss_fn(pred[:, :N_STAGE], obs, mask, W_S, lam_p, lam_b,
                        C_in_meas, Cg_inlet)                  # dict(total,data,phys,bc)
        terms['total'].backward()
        torch.nn.utils.clip_grad_norm_(func.parameters(), 1.0)   # 발산 대비(B.4)
        opt.step()
        if epoch % 20 == 0:
            print(f"  epoch {epoch:4d}  loss={float(terms['total']):.4e}")
    return func


def compare_table4(func):
    """B.5 — 추정값 vs 논문 Table 4."""
    est = dict(k1=float(func.log_k1.exp()), a1=float(func.a1), b1=float(func.b1),
               k2=float(func.log_k2.exp()), a2=float(func.a2), b2=float(func.b2))
    paper = dict(k1=500.6, a1=1.0, b1=1005.5, k2=4380.0, a2=1.0, b2=1263.0)
    print("\n  파라미터   논문값      추정값      상대오차")
    for key in ['k1', 'a1', 'b1', 'k2', 'a2', 'b2']:
        p, e = paper[key], est[key]
        rel = abs(e - p) / abs(p) if p else float('nan')
        print(f"  {key:6s}   {p:9.3f}   {e:9.3f}   {rel*100:6.1f}%")


def main():
    data_paths = sorted(glob.glob(os.path.join(M.REPO, 'data', 'withLabel', '1*.xlsx')))
    print("=" * 74)
    print(" [Week-2] Grey-box 파라미터 추정 (torchdiffeq)")
    print("=" * 74)
    try:
        prep = M.load_inputs(data_paths[2])   # (실제로는 Set-3 정합 필요)
        func = estimate(prep)
        compare_table4(func)
    except NotImplementedError as e:
        print("\n[!] 물리/loss 코어가 아직 비어 있습니다 (정상 — 직접 구현 대상):")
        print("    ->", e)
        print("    rates(), forward(), loss_fn(), load_targets() 구현 후 실행.")


if __name__ == "__main__":
    main()
