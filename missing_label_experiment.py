# -*- coding: utf-8 -*-
"""
==============================================================================
 missing_label_experiment.py  --  [Week-4 / 산출물 E] 결측 라벨 실험
------------------------------------------------------------------------------
 프로젝트의 헤드라인 주장 입증: "물리가 결측 라벨을 메운다".
   라벨을 점점 지워도(hybrid) 완만히 degrade, 순수 데이터 모델(LSTM/SSAE)은 급락.
   -> RMSE vs 라벨비율 곡선 하나로 novelty 를 보인다.

 핵심 메커니즘 (wekk3-4.md E.2):
   라벨 제거는 L_data 에서만 빠진다. L_phys 는 라벨 불필요한 collocation 점에서
   계속 계산되므로 hybrid 는 학습 신호가 남는다. 데이터 모델은 그 지점서 0 -> 급락.

 ┌───────────────────────────────────────────────────────────────────┐
 │  ★ 주의 (wekk3-4.md E.5): 마스크는 L_data 에만 적용.               │
 │     물리·경계 항에는 절대 마스크를 걸지 말 것 — 이 실험 성립의 이유.│
 │  ★ 구조적(structured) 마스킹 로직은 학습 핵심이므로 TODO.          │
 └───────────────────────────────────────────────────────────────────┘
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mechanistic_model as M

N_STAGE = M.N_STAGE


# ==========================================================================
# full_mask 구성 (plumbing) — 실제 측정 라벨 위치 -> (T, N_STAGE) bool
# ==========================================================================
def build_full_mask(prep):
    """
    prep['label'][t] == s(1..6) 이면 그 시점 sampling point s 가 실제 측정됨.
      stage 1..5 만 L_data 대상 (point6=inlet 은 L_bc 경계항 -> 제외).
    반환: torch.bool (T, N_STAGE). 이 mask 를 make_label_mask 로 축소해 사용.
    """
    label = np.asarray(prep['label']).astype(int)
    T = prep['n']
    mask = np.zeros((T, N_STAGE), dtype=bool)
    for t in range(T):
        s = label[t]
        if 1 <= s <= N_STAGE:
            mask[t, s - 1] = True
    return torch.as_tensor(mask)


# ==========================================================================
# 라벨 마스크 생성 (E.3) — 랜덤은 제공, 구조적은 직접 구현
# ==========================================================================
def make_label_mask(full_mask, keep_fraction, strategy='random', seed=0):
    """
    full_mask   : (T, n_stages) bool — 원래 라벨이 존재하는 위치
    keep_fraction : 유지할 라벨 비율 (예: 0.2 => 20% 만 남김)
    strategy    : 'random'(MCAR) | 'structured'(위치/시간블록 통째 제거)
    반환        : 축소된 bool 마스크 (L_data 에만 사용)
    """
    g = torch.Generator().manual_seed(seed)
    if strategy == 'random':
        # MCAR: 균일 확률로 유지
        keep = torch.rand(full_mask.shape, generator=g) < keep_fraction
        return full_mask & keep
    elif strategy == 'structured':
        # 회전 분석기 패턴 모사: 연속 '시간블록'을 통째로 제거 (E.3).
        #   시간축을 블록으로 나눠, keep_fraction 비율만큼 블록만 남기고
        #   나머지 블록 구간의 라벨을 통째로 제거 (랜덤 산발 제거보다 어렵고 현실적).
        T = full_mask.shape[0]
        n_blocks = 8
        block_len = max(1, T // n_blocks)
        block_id = (torch.arange(T) // block_len)
        n_bid = int(block_id.max().item()) + 1
        perm = torch.randperm(n_bid, generator=g)
        n_keep = max(1, int(round(keep_fraction * n_bid)))
        keep_blocks = set(perm[:n_keep].tolist())
        keep_time = torch.tensor([int(b) in keep_blocks for b in block_id])   # (T,)
        m = full_mask.clone()
        m[~keep_time] = False                     # 제거 블록 구간은 라벨 통째 제거
        return m
    else:
        raise ValueError(f"unknown strategy: {strategy}")


# ==========================================================================
# 스윕 실험 (plumbing 골격) — E.4 핵심 그림 데이터 생성
# ==========================================================================
KEEP_FRACTIONS = [1.00, 0.75, 0.50, 0.35, 0.20]
MODELS = ['hybrid', 'greybox', 'lstm', 'ssae']
SEEDS = [0, 1, 2]                    # 시드 3개↑ 평균±표준편차 (E.4)


def run_sweep(prep, full_mask, train_model_fn, strategies=('random', 'structured')):
    """
    라벨비율 × 모델 × 시드 스윕 -> {(strategy, model, kf): [rmse per seed]}.
    train_model_fn(model_name, prep, mask, seed) -> test_rmse  (학생이 연결)

    ※ 마스크는 L_data 에만. L_phys/L_bc 는 항상 full collocation 유지.
    """
    results = {}
    for strat in strategies:
        for kf in KEEP_FRACTIONS:
            for model in MODELS:
                rmses = []
                for seed in SEEDS:
                    if model in ('hybrid', 'greybox'):
                        mask = make_label_mask(full_mask, kf, strat, seed)
                    else:
                        # 순수 데이터 모델: 마스크된 라벨만 학습에 사용 (collocation 신호 없음)
                        mask = make_label_mask(full_mask, kf, strat, seed)
                    # [TODO] 각 모델 학습/평가를 연결 (hybrid_ude.train / reproduce.trainSequenceLSTM 등)
                    rmse = train_model_fn(model, prep, mask, seed)   # 학생이 구현/연결
                    rmses.append(rmse)
                results[(strat, model, kf)] = rmses
    return results


def _curve(results, strat, model):
    """(keep%, mean, std) 배열 — 시드 평균."""
    xs, ys, es = [], [], []
    for kf in sorted(KEEP_FRACTIONS, reverse=True):
        vals = [r for r in results.get((strat, model, kf), []) if r is not None]
        if vals:
            a = np.array(vals, dtype=float)
            xs.append(kf * 100); ys.append(a.mean()); es.append(a.std())
    return np.array(xs), np.array(ys), np.array(es)


def summarize(results, strategy='random'):
    """평균±표준편차 표 + degradation 기울기 + 교차점 (E.4/E.6). (plumbing)"""
    print(f"\n  [{strategy}]  model     keep%   RMSE(mean±std)")
    for model in MODELS:
        x, y, e = _curve(results, strategy, model)
        for xi, yi, ei in zip(x, y, e):
            print(f"    {model:8s} {int(xi):4d}   {yi:.3f} ± {ei:.3f}")

    # degradation 기울기: 라벨 100%->20% 로 줄 때 RMSE 상승량 (작을수록 강건)
    print("\n  degradation 기울기 (Δ RMSE, 100%->20%; 작을수록 물리가 잘 메움):")
    slopes = {}
    for model in MODELS:
        x, y, _ = _curve(results, strategy, model)
        if x.size >= 2:
            slopes[model] = y[-1] - y[0]           # 낮은 keep% - 높은 keep%
            print(f"    {model:8s} : {slopes[model]:+.3f}")
    if slopes:
        best = min(slopes, key=slopes.get)
        print(f"  -> 가장 평탄한 모델: {best} (통과기준 E.6: hybrid 여야 함)")

    # 교차점: hybrid 가 baseline(lstm) 아래로 내려가는 keep% (헤드라인)
    xh, yh, _ = _curve(results, strategy, 'hybrid')
    xl, yl, _ = _curve(results, strategy, 'lstm')
    if xh.size and xl.size:
        common = sorted(set(xh) & set(xl), reverse=True)
        cross = next((k for k in common
                      if yh[list(xh).index(k)] < yl[list(xl).index(k)]), None)
        print(f"  hybrid<LSTM 교차점: {('keep '+str(int(cross))+'%' if cross else '없음(추가 실험 필요)')}")


def plot_label_curve(results, strategy='random', out=None):
    """RMSE vs 라벨비율 곡선 (E.4 핵심 그림). (plumbing, 즉시 실행 가능)"""
    out = out or f'missing_label_{strategy}.png'
    colors = dict(hybrid='tab:red', greybox='tab:orange', lstm='tab:blue', ssae='tab:green')
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODELS:
        x, y, e = _curve(results, strategy, model)
        if x.size:
            ax.errorbar(x, y, yerr=e, marker='o', capsize=3,
                        color=colors.get(model), label=model)
    ax.set_xlabel('label kept (%)'); ax.set_ylabel('Test RMSE')
    ax.set_title(f'Missing-label experiment ({strategy})')
    ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  곡선 저장: {out}")


def main():
    print("=" * 74)
    print(" [Week-4] 결측 라벨 실험 (RMSE vs 라벨비율 곡선)")
    print("=" * 74)
    print(" 통과기준(E.6):")
    print("  - 20% 라벨에서 hybrid < LSTM/SSAE (명확히)")
    print("  - hybrid 의 라벨비율-RMSE 곡선이 가장 평탄(기울기 최소)")
    print("  - 완전 라벨에서 hybrid ≈ 또는 < fused(0.123)")
    print("\n 연결 필요: full_mask 구성 + train_model_fn(각 모델 학습/평가) + structured 마스킹.")


if __name__ == "__main__":
    main()
