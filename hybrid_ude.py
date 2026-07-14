# -*- coding: utf-8 -*-
"""
==============================================================================
 hybrid_ude.py  --  [Week-3 / 산출물 C] Hybrid UDE (NN 닫힘항)
------------------------------------------------------------------------------
 GreyBoxODE 에서 rates() 만 NNClosure(Rθ) 로 교체한 구조.
 balance(A.1) / 경계 / 적분 / loss_fn(B) 은 2주차 것을 '그대로' 재사용
 -> 바뀌는 건 반응항 한 곳뿐 (버그 표면적 최소, wekk3-4.md C.0).

 ┌───────────────────────────────────────────────────────────────────┐
 │  ★ 무결성 노트 (wekk3-4.md)                                         │
 │  forward() 의 balance 조립은 2주차 greybox_ode.forward() 와 동일    │
 │  물리 -> 그 구현을 재사용/이식하는 것도 학습의 일부이므로 TODO.     │
 │  warm-start distill, 물리제약, adaptive weighting 핵심도 TODO.      │
 └───────────────────────────────────────────────────────────────────┘
"""
import os
import glob
import numpy as np

import torch
import torch.nn as nn
from torchdiffeq import odeint

import mechanistic_model as M
import greybox_ode as G                 # loss_fn, make_inputs_fn, load_targets 재사용
import loss_tuning as LT                # per_term_grad_norms, grad_ratio (진단 plumbing)
from nn_closure import NNClosure, distill_from_greybox

torch.set_default_dtype(torch.float64)
N_STAGE = M.N_STAGE


class HybridUDE(nn.Module):
    def __init__(self, consts, inputs_fn, closure):
        super().__init__()
        self.consts = consts
        self.inputs_fn = inputs_fn
        self.closure = closure          # NNClosure (Rθ)

    def forward(self, t, x):
        Cg, Cl = x[..., :N_STAGE], x[..., N_STAGE:2 * N_STAGE]
        Fg, Fl, C_in, T = self.inputs_fn(t)
        R = self.closure(Cg, Cl, T)     # ★ Arrhenius -> Rθ 로 교체된 유일한 지점
        # balance 는 greybox_ode.forward 와 '동일' (C.0: 반응항만 교체).
        eps_l, eps_g, V = self.consts['EPS_L'], self.consts['EPS_G'], self.consts['V_SEC']
        Cg_in = torch.cat([Cg[1:], C_in.reshape(1)])                    # 기체: 아래단(i+1)/inlet
        Cl_in = torch.cat([torch.zeros(1, dtype=Cl.dtype), Cl[:-1]])    # 액체: 위단(i-1)/lean
        dCg = (Fg * (Cg_in - Cg) / V - eps_l * R) / eps_g              # Eq.10
        dCl = Fl * (Cl_in - Cl) / (V * eps_l) + R                       # Eq.11
        return torch.cat([dCg, dCl])


# ==========================================================================
# 학습 루프 (plumbing 제공) — warm-start 후 joint fine-tune
#  + Loss 가중치 튜닝 로깅 (wekk3-4.md D: 항별 loss/grad-norm)
# ==========================================================================
def train(prep, epochs=400, lr=1e-3, lam_p=1.0, lam_b=1.0,
          warm_start=True, log_grad_balance=True):
    inputs_fn, t_grid = G.make_inputs_fn(prep)
    consts = dict(EPS_L=M.EPS_L, EPS_G=M.EPS_G, V_SEC=M.V_SEC,
                  EA1=M.EA1, EA2=M.EA2, RGAS=M.RGAS)

    closure = NNClosure(hidden=32)
    if warm_start:
        # C.2 warm-start: 기준 Arrhenius(Table 4 값으로 세팅한 grey-box)를 Rθ 에 distill.
        #   (2주차에서 실제 피팅한 grey-box 가 있으면 그 func.rates 를 넘기면 됨)
        ref = G.GreyBoxODE(consts, inputs_fn)
        with torch.no_grad():
            ref.log_k1.fill_(float(np.log(M.K1))); ref.log_k2.fill_(float(np.log(M.K2)))
            ref.a1.fill_(M.ALPHA1); ref.a2.fill_(M.ALPHA2)
            ref.b1.fill_(M.BETA1); ref.b2.fill_(M.BETA2)
        print("  [warm-start] 기준 Arrhenius(Table 4) -> Rθ distill")
        distill_from_greybox(closure, ref.rates, epochs=300)

    func = HybridUDE(consts, inputs_fn, closure)
    opt = torch.optim.Adam(func.parameters(), lr=lr)
    params = list(func.parameters())
    x0 = torch.zeros(2 * N_STAGE)
    obs, mask, C_in_meas = G.load_targets(prep)      # TODO(2주차) 완성 필요

    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        pred = odeint(func, x0, t_grid, method='dopri5')
        Cg_inlet = pred[:, N_STAGE - 1]

        # loss_fn 은 dict(total,data,phys,bc) 반환 (greybox_ode.loss_fn 계약)
        terms = G.loss_fn(pred[:, :N_STAGE], obs, mask, G.W_S, lam_p, lam_b,
                          C_in_meas, Cg_inlet)

        # --- D 진단(plumbing): 항별 grad norm 측정 ---
        gnorms = LT.per_term_grad_norms(terms, params)   # {'data','phys','bc'}
        ratio = LT.grad_ratio(gnorms)                    # >10 이면 불균형

        # ============================================================
        # ★★★ 학습 코어 (직접 구현): adaptive weighting 규칙 ★★★
        #   gnorms 를 보고 lam_p, lam_b 를 갱신 (wekk3-4.md D.2 ③~⑤).
        #   갱신된 λ 는 다음 epoch 의 loss_fn 호출에 반영된다.
        #   예) LR-anneal(Wang):  lam_p = gnorms['data'] / (gnorms['phys'] + 1e-8)
        #   if adaptive:
        #       lam_p, lam_b = <여기에 직접 구현>
        # ============================================================

        terms['total'].backward()
        gtot = torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        if epoch % 20 == 0:
            rec = dict(epoch=epoch,
                       data=float(terms['data']), phys=float(terms['phys']), bc=float(terms['bc']),
                       gn_data=gnorms['data'], gn_phys=gnorms['phys'], gn_bc=gnorms['bc'],
                       ratio=ratio)
            history.append(rec)
            print(f"  epoch {epoch:4d} L={float(terms['total']):.3e} "
                  f"gn(d/p/b)={gnorms['data']:.1e}/{gnorms['phys']:.1e}/{gnorms['bc']:.1e} "
                  f"ratio={ratio:.1f}")
    return func, history


def main():
    data_paths = sorted(glob.glob(os.path.join(M.REPO, 'data', 'withLabel', '1*.xlsx')))
    print("=" * 74)
    print(" [Week-3] Hybrid UDE (NN 닫힘항 Rθ) 학습")
    print("=" * 74)
    try:
        prep = M.load_inputs(data_paths[2])
        func, hist = train(prep)
        print(" 완료. 통과기준(C.4): hybrid RMSE <= grey-box, baseline 대비 보고.")
    except NotImplementedError as e:
        print("\n[!] 물리/loss/warm-start 코어가 아직 비어 있습니다 (정상 — 직접 구현):")
        print("    ->", e)
        print("    2주차(greybox_ode: forward/loss/load_targets) + 3주차(NNClosure 제약,")
        print("    distill, HybridUDE.forward) 구현 후 실행하세요.")


if __name__ == "__main__":
    main()
