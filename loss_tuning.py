# -*- coding: utf-8 -*-
"""
==============================================================================
 loss_tuning.py  --  [산출물 D] Loss 가중치 튜닝 plumbing (비-코어)
------------------------------------------------------------------------------
 여기 있는 것(전부 즉시 사용 가능한 plumbing):
   - per_term_grad_norms : 항별(L_data/L_phys/L_bc) gradient norm 측정 (진단)
   - grad_ratio          : 항간 불균형 비율 (>10 이면 조치 필요)
   - lambda_grid         : 고정 그리드 λ 스윕 조합
   - LossTracker         : run 별 결과를 CSV/Markdown 트래커로 기록
   - plot_history        : epoch 별 항별 loss·grad norm 곡선 그림

 ┌───────────────────────────────────────────────────────────────────┐
 │  ★ 학습 코어(직접 구현)는 여기 없다. 코어 위치:                     │
 │   1) greybox_ode.loss_fn : dict(total,data,phys,bc) 반환 구현       │
 │   2) hybrid_ude.train    : "adaptive weighting 규칙" (λ 자동 갱신)  │
 │      -> LR-anneal / uncertainty / 커리큘럼 중 택해 직접 구현        │
 └───────────────────────────────────────────────────────────────────┘
"""
import csv
import os
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================================
# 진단: 항별 gradient norm (loss_fn 이 dict(total,data,phys,bc) 반환한다고 가정)
# ==========================================================================
def per_term_grad_norms(loss_terms, params):
    """
    loss_terms : dict — 'data','phys','bc' 키에 각 항 텐서(스칼라).
    params     : 모델 파라미터 리스트.
    반환       : {'data':‖∂L_data/∂θ‖, 'phys':..., 'bc':...}
    ※ 실제 backward 전에 호출. retain_graph=True 로 그래프 보존.
    """
    import torch
    out = {}
    params = list(params)
    for key in ('data', 'phys', 'bc'):
        if key not in loss_terms or loss_terms[key] is None:
            out[key] = float('nan'); continue
        g = torch.autograd.grad(loss_terms[key], params,
                                retain_graph=True, allow_unused=True)
        sq = sum((x ** 2).sum() for x in g if x is not None)
        out[key] = float(torch.sqrt(sq)) if sq is not None else 0.0
    return out


def grad_ratio(gnorms):
    """항간 최대/최소 grad norm 비율. >10 이면 불균형(조치 필요)."""
    vals = [v for v in gnorms.values() if v and np.isfinite(v) and v > 0]
    return max(vals) / min(vals) if len(vals) > 1 else float('nan')


def lambda_grid(values=(0.01, 0.1, 1.0, 10.0)):
    """(λ_p, λ_b) 고정 그리드 조합 (D.2 방법①)."""
    return list(itertools.product(values, values))


# ==========================================================================
# 트래커: run 별 결과 -> CSV + Markdown (LOSS_TUNING.md 표 채우기)
# ==========================================================================
class LossTracker:
    FIELDS = ['run', 'lam_p', 'lam_b', 'weighting', 'seed',
              'val_rmse', 'test_rmse', 'grad_ratio', 'balance', 'note']

    def __init__(self, csv_path='loss_tuning_log.csv'):
        self.csv_path = csv_path
        self.rows = []

    def log(self, run, lam_p, lam_b, weighting, seed,
            val_rmse=None, test_rmse=None, gratio=None, note=''):
        bal = ('OK(≤10)' if (gratio is not None and np.isfinite(gratio) and gratio <= 10)
               else ('불균형(>10)' if gratio and np.isfinite(gratio) else '-'))
        self.rows.append(dict(run=run, lam_p=lam_p, lam_b=lam_b, weighting=weighting,
                              seed=seed, val_rmse=val_rmse, test_rmse=test_rmse,
                              grad_ratio=(round(gratio, 2) if gratio and np.isfinite(gratio) else None),
                              balance=bal, note=note))

    def save_csv(self):
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader(); w.writerows(self.rows)
        print(f"  트래커 저장: {self.csv_path}")

    def to_markdown(self, path='loss_tuning_result.md'):
        def fmt(x):
            return '—' if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
        lines = ["| " + " | ".join(self.FIELDS) + " |",
                 "|" + "|".join(["---"] * len(self.FIELDS)) + "|"]
        for r in self.rows:
            lines.append("| " + " | ".join(fmt(r[k]) for k in self.FIELDS) + " |")
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Markdown 표 저장: {path}")


# ==========================================================================
# 그림: epoch 별 항별 loss·grad norm (진단 시각화)
# ==========================================================================
def plot_history(history, out='loss_tuning_history.png'):
    """
    history : list of dict, 각 원소 예:
      {'epoch':e, 'data':.., 'phys':.., 'bc':.., 'gn_data':.., 'gn_phys':.., 'gn_bc':..}
    """
    if not history:
        print("  (history 비어있음 — 학습 루프에서 항별 로깅을 채운 뒤 호출)"); return
    ep = [h['epoch'] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, c in [('data', 'tab:blue'), ('phys', 'tab:red'), ('bc', 'tab:green')]:
        if key in history[0]:
            ax[0].plot(ep, [h[key] for h in history], color=c, label=f'L_{key}')
        gk = 'gn_' + key
        if gk in history[0]:
            ax[1].plot(ep, [h[gk] for h in history], color=c, label=f'|∇L_{key}|')
    ax[0].set_title('항별 loss'); ax[0].set_xlabel('epoch'); ax[0].set_yscale('log'); ax[0].legend()
    ax[1].set_title('항별 gradient norm'); ax[1].set_xlabel('epoch'); ax[1].set_yscale('log'); ax[1].legend()
    ax[1].axhline(0, color='gray', lw=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  그림 저장: {out}")


if __name__ == "__main__":
    # 데모: 트래커/그리드가 실제로 동작함을 보임 (코어 없이도 plumbing 확인)
    print("lambda_grid(4값) 조합 수:", len(lambda_grid()))
    tr = LossTracker()
    tr.log('d01', 1, 1, 'fixed', 0, val_rmse=0.21, test_rmse=0.24, gratio=15.0, note='baseline')
    tr.log('d07', None, None, 'LR-anneal', 0, val_rmse=0.19, test_rmse=0.20, gratio=3.2)
    tr.save_csv(); tr.to_markdown()
    print("plumbing OK — 학습 코어(loss_fn dict 반환 / adaptive 규칙)만 채우면 연결됨.")
