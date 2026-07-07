# -*- coding: utf-8 -*-
"""
==============================================================================
 make_fig11.py  --  논문 Fig.11 재현 (우리 재현 결과로 그림 생성)
------------------------------------------------------------------------------
 Fig.11 (Zhuang et al., 2022):
   (a) Set-1 의 sampling point 5,6   (b) Set-2 의 sampling point 5,6
   (c) sampling point 1-4
   ✕ = 가스분석기 실측값,  선 = Mechanistic / DAE-LSTM / Fused / SSAE 추정

 입력: reproduce.py 가 저장한 fig11_set{1,2}_{meas,mech,daelstm,fused}.npy
       + results/set{1,2}_{1..6}.npy (원저자 SSAE)
 출력: fig11_repro.png
==============================================================================
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reproduce as R   # 모듈 임포트 시 os.chdir(repo root) 수행됨

CB = R.CALLBACK

# 테스트 시나리오 -> 원본 df 인덱스 (reproduce.TEST_INDEX_MAP)
SET_TEST_IDX = {"1": 2, "2": 1}   # Set-1=140207_1(idx2), Set-2=140206_1(idx1)


def load_arrays(tag):
    return dict(
        meas=np.load(f"fig11_{tag}_meas.npy"),
        mech=np.load(f"fig11_{tag}_mech.npy"),
        dae=np.load(f"fig11_{tag}_daelstm.npy"),
        fused=np.load(f"fig11_{tag}_fused.npy"),
    )


def discrete_high(df):
    """지점 5,6 실측 ✕: 원 AT400 기록에서 0.5% 초과값만."""
    rec = df["AT400(CO2 %)"].values[CB:].astype(float)
    return np.where(rec > 0.5, rec, np.nan)


def discrete_low(df):
    """지점 1-4 실측 ✕: 각 시점 label 위치에만 값 배치 (notebook columnSeparator2)."""
    d = R.avgOutPoint1(df.set_index("time").copy())
    lab = d["label"].values
    at = d["AT400(CO2 %)"].values
    n = d.shape[0]
    arr = np.full((n, 4), np.nan)
    for i in range(n):
        L = int(lab[i])
        if L <= 4:
            arr[i, L - 1] = at[i]
    return arr[CB:]


def ssae_points(setnum, pts):
    out = {}
    for p in pts:
        f = f"results/set{setnum}_{p}.npy"
        out[p] = np.load(f).ravel() if os.path.exists(f) else None
    return out


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


def main():
    origin = R.load_all_dataframes(sorted(glob.glob("data/withLabel/1*.xlsx")))

    A1 = load_arrays("set1"); A2 = load_arrays("set2")
    df1 = origin[SET_TEST_IDX["1"]]; df2 = origin[SET_TEST_IDX["2"]]
    ss1 = ssae_points(1, [1, 2, 3, 4, 5, 6])
    ss2 = ssae_points(2, [1, 2, 3, 4, 5, 6])

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    plt.subplots_adjust(wspace=0.18, bottom=0.15)

    # ---- (a) Set-1, 지점 6,5 ----
    def panel_top(ax, A, df, ss, title):
        cr = discrete_high(df)
        x = range(A["meas"].shape[0])
        ax.scatter(x, cr[:len(x)], marker="x", color="black", s=28, label="Measurements", zorder=5)
        for j, c in [(5, "tab:blue"), (4, "tab:red")]:      # point6=idx5, point5=idx4
            ax.plot(A["mech"][:, j], ls=":", color=c, label=f"Mechanistic @{j+1}")
            ax.plot(A["dae"][:, j], ls="--", color=c, label=f"DAE-LSTM @{j+1}")
            ax.plot(A["fused"][:, j], ls="-", color=c, lw=1.8, label=f"Fused @{j+1}")
            if ss.get(j + 1) is not None:
                s = ss[j + 1]
                ax.plot(range(len(s)), s, ls="-.", color=c, alpha=0.6, label=f"SSAE @{j+1}")
        ax.set_title(title); ax.set_xlabel("Time"); ax.set_ylabel("CO$_2$ (%)")

    panel_top(axs[0], A1, df1, ss1, "(a) Set-1  points 5,6")
    panel_top(axs[1], A2, df2, ss2, "(b) Set-2  points 5,6")
    axs[0].legend(fontsize=6, ncol=2, loc="upper right")

    # ---- (c) Set-1, 지점 1-4 ----
    ax = axs[2]
    low = discrete_low(df1)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    mk = ["o", "x", "s", "^"]
    for p in range(4):
        ax.scatter(range(low.shape[0]), low[:, p], marker=mk[p], s=18, color=colors[p],
                   label=f"Meas @{p+1}")
        ax.plot(A1["fused"][:, p], ls="--", color=colors[p])
        if ss1.get(p + 1) is not None:
            ax.plot(range(len(ss1[p + 1])), ss1[p + 1], ls="-", color=colors[p], alpha=0.6)
    ax.set_title("(c) Set-1  points 1-4  (dashed=Fused, solid=SSAE)")
    ax.set_xlabel("Time"); ax.set_ylabel("CO$_2$ (%)")
    ax.legend(fontsize=6, ncol=2, loc="upper right")

    fig.suptitle("Fig.11 Reproduction - Measurements(x) vs Mechanistic / DAE-LSTM / Fused / SSAE", y=1.02)
    fig.savefig("fig11_repro.png", dpi=150, bbox_inches="tight")
    print("saved fig11_repro.png")

    # 재현 RMSE 재확인 (배열로부터)
    for tag, A in [("Set-1", A1), ("Set-2", A2)]:
        print(f"[{tag}] Mechanistic={rmse(A['mech'], A['meas']):.3f}  "
              f"DAE-LSTM={rmse(A['dae'], A['meas']):.3f}  "
              f"Fused={rmse(A['fused'], A['meas']):.3f}")


if __name__ == "__main__":
    main()
