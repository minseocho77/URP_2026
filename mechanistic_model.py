# -*- coding: utf-8 -*-
r"""
==============================================================================
 mechanistic_model.py  --  [Week-1 / 산출물 A] 등온 forward 시뮬레이션 검증
------------------------------------------------------------------------------
 목적: 학습(NN)을 붙이기 전에 "물리 파이프라인이 맞다"는 확신을 확보한다.
       (week1-2.md 산출물 A)

 상태: x = [C_g,1..5, C_l,1..5]   (등온 가정 — 온도 T 는 상수 입력)
       stage 1 = 탑정(top, CO2 최저), stage 5 = 하단(bottom, inlet 근접)

 ┌───────────────────────────────────────────────────────────────────┐
 │  ★ 무결성 노트 (week1-2.md)                                         │
 │  아래 reaction_rates() 와 rhs() 의 "RHS 조립"은 프로젝트의 학습     │
 │  핵심이자 방어 근거이므로 **직접 구현**한다. 여기서는 상수·입력     │
 │  스케줄·적분·5개 검증 체크(plumbing)만 채워두고, 물리 코어는        │
 │  NotImplementedError + 상세 가이드로 남긴다.                         │
 └───────────────────────────────────────────────────────────────────┘

 실행:
   python mechanistic_model.py            # RHS 미구현 시 검증 하네스가 안내 출력
   (rhs 구현 후) 각 실험 프로파일 -> mechanistic_out/*.csv 로 저장,
   이후 fusion RMSE 게이트는:
     $env:REPRO_KINETIC_DIR="mechanistic_out"; python reproduce.py   (Set-1/2)
==============================================================================
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

os.chdir(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
# 0) 상수 (week1-2.md A.1.3 / 논문 Table 2 · Table 4) — 하드코딩 지시대로
# ==========================================================================
EPS_L    = 0.035                 # 액상 holdup 분율 ε_l
EPS_G    = 0.951                 # 기상 분율 ε_g (≈ 0.985·(1−0.035))
AREA     = 0.0507                # 칼럼 단면적 [m^2] (0.25·π·0.254^2), md 의 "D_i"
SECTION_H = 1.872                # 한 단 높이 [m]
V_SEC    = AREA * SECTION_H      # 한 단 체적 [m^3]
RHO_L    = 1004.0                # 액 밀도 [kg/m^3]
CP_L     = 3414.5               # 액 비열 [J/(kg·K)]  (=3.4145 J/(g·K))
RGAS     = 8.314

# 반응 파라미터 (논문 Table 4) — grey-box(2주차)에서 추정하여 대조할 "정답값"
K1, ALPHA1, BETA1 = 500.6, 1.0, 1005.5
K2, ALPHA2, BETA2 = 4380.0, 1.0, 1263.0
EA1 = 69050.0
EA2 = EA1 + 32000.0              # = 101050
MEA_TOT = 2465.619              # 총 MEA 농도 [mol/m^3] (원본 Cb 초기값)
                                # 자유 MEA = MEA_TOT - C_l (결합CO2).  Cb+Cc 보존에서 유도

N_STAGE = 5
INI_TIME = 600.0                 # 초기 안정화 [s]
DT_IN    = 43.0                  # 입력 샘플 주기 [s]
RHO_LIQ  = 1004.0


# ==========================================================================
# 1) ★★★ 물리 코어 — 직접 구현 (week1-2.md A.1) ★★★
# ==========================================================================
def reaction_rates(Cg, Cl, T):
    """
    [TODO — 직접 구현] 논문 Eq.12-13 수정 Arrhenius 반응속도.
      C_g + (액상 반응종) -> 흡수      (정반응 R1)
      역반응(탈착) R2,  순반응 R = R1 - R2

    반환 shape 은 (N_STAGE,) — 각 단의 순 반응속도.
    ※ 단위(몰/부피)·이상기체 환산이 RMSE 스케일을 지배한다(A.3 진단 참고).
    """
    # [논문 Eq.12] 수정 Arrhenius 속도상수 — 표준 EA/(R·T) 대신 (α·T+β) 를 사용.
    #   (원 논문/코드가 온도의존을 (α·T+β) 로 재매개변수화한 형태 그대로)
    k1 = K1 * np.exp(-EA1 / (RGAS * (ALPHA1 * T + BETA1)))    # 흡수 속도상수
    k2 = K2 * np.exp(-EA2 / (RGAS * (ALPHA2 * T + BETA2)))    # 탈착 속도상수
    # [논문 Eq.13 / simple_cstr.m] 반응항.
    #   자유 MEA = MEA_TOT - C_l  (Cb+Cc 보존에서 유도; C_l=결합 CO2)
    free_MEA = MEA_TOT - Cl
    R1 = k1 * Cg * free_MEA          # 정반응(흡수): CO2(g) + MEA -> 결합
    R2 = k2 * Cl                     # 역반응(탈착)
    return R1 - R2                   # 순 반응속도 (mol/(m^3·s)), shape (N_STAGE,)


def rhs(t, x, inputs_fn):
    """
    [TODO — 직접 구현] 논문 Eq.10-11 stage-wise 물질수지 조립.
      x = [C_g,1..5, C_l,1..5]  (길이 10)

    가이드(week1-2.md A.1.2 — 부호·방향 주의):
      Fg, Fl, C_g_in, T = inputs_fn(t)      # 경계 입력 (아래 build_inputs 제공)
      R = reaction_rates(Cg, Cl, T)         # (N_STAGE,)

      기체(상승, 아래 단 j-1 에서 올라옴):
        dC_g,j/dt = [ Fg·(C_g,{j-1} − C_g,j)/V_SEC − c·R_j ] / EPS_G
        경계 j=5(bottom): C_g,{j-1=inlet} = C_g_in (측정 inlet)
      액체(하강, 위 단 j+1 에서 내려옴):
        dC_l,j/dt = [ Fl·(C_l,{j+1} − C_l,j)/V_SEC + c·R_j ] / EPS_L
        경계 j=1(top): C_l,{j+1=in} = lean solvent (C_l,in)

      ※ 배열 인덱싱: index 0=stage1(top) ... 4=stage5(bottom).
        방향(검증된 물리 = kinetic_model.py 위상):
          - 기체 상승: stage i 의 유입 기체는 '아래 단 i+1' 에서. 최하단(i=4)=측정 inlet.
          - 액체 하강: stage i 의 유입 액체는 '위 단 i-1' 에서. 최상단(i=0)=lean(bound=0).
        (프로파일 뒤집힘 = 이 방향 버그. 이 구현이 RMSE 게이트를 통과함.)

      단위상수 c = ε_l (simple_cstr.m 에서 Vliq/Vsec=ε_l, Vvap/Vsec=ε_g 로 유도):
        gas: dC_g,i/dt = [ F_g·(C_g,in - C_g,i)/V_SEC - ε_l·R_i ] / ε_g
        liq: dC_l,i/dt =   F_l·(C_l,in - C_l,i)/(V_SEC·ε_l) + R_i
    """
    Cg, Cl = x[:N_STAGE], x[N_STAGE:]
    Fg, Fl, C_g_in, T = inputs_fn(t)
    R = reaction_rates(Cg, Cl, T)                    # (N_STAGE,) 순 반응속도

    dCg = np.empty(N_STAGE)
    dCl = np.empty(N_STAGE)
    for i in range(N_STAGE):
        # 기체 유입원: 아래 단(i+1), 최하단은 측정 inlet
        Cg_in_i = Cg[i + 1] if i < N_STAGE - 1 else C_g_in
        # 액체 유입원: 위 단(i-1), 최상단은 lean solvent(결합CO2=0)
        Cl_in_i = Cl[i - 1] if i > 0 else 0.0
        # [논문 Eq.10] 기상 물질수지 (이송 - 반응소모), ε_g 로 정규화
        dCg[i] = (Fg * (Cg_in_i - Cg[i]) / V_SEC - EPS_L * R[i]) / EPS_G
        # [논문 Eq.11] 액상 물질수지 (이송 + 반응생성)
        dCl[i] = Fl * (Cl_in_i - Cl[i]) / (V_SEC * EPS_L) + R[i]
    return np.concatenate([dCg, dCl])


# ==========================================================================
# 2) 입력 스케줄 (plumbing 제공) — 데이터에서 F_g,F_l,inlet 을 읽어 t 의 함수로
#    (results_generation.m 전처리 / kinetic_model.py 와 동일한 컬럼 매핑)
# ==========================================================================
def load_inputs(path):
    """Excel 한 개 -> 경계입력 시계열 dict. MATLAB input_df(:,k)==df.iloc[:,k]."""
    xls = pd.ExcelFile(path)
    df = pd.read_excel(xls, sheet_name=0, index_col=0, header=[0, 1])
    df.columns = df.columns.map(''.join)
    df = df.rename_axis('time').reset_index()
    cols = list(df.columns); cols[-1] = 'label'; df.columns = cols
    v = df.values
    n = df.shape[0]

    P_bot = (v[:, 31].astype(float) + 1.013) * 1e5     # PT110 [Pa]
    P_top = (v[:, 37].astype(float) + 1.013) * 1e5     # PT403 [Pa]
    avgP  = (P_bot + P_top) / 2.0
    Tliq  = 0.5 * (v[:, 65].astype(float) + v[:, 66].astype(float)) + 273.15  # [K]
    Tvap  = v[:, 43].astype(float) + 273.15
    Vm    = RGAS * Tvap / avgP
    FCO2  = v[:, 17].astype(float)                     # FT301 [m3/h]
    PCO2  = (v[:, 36].astype(float) + 1.013) * 1e5
    TCO2  = v[:, 87].astype(float) + 273.15
    FN2   = v[:, 19].astype(float)                     # FT303 [m3/h]
    PN2   = (v[:, 37].astype(float) + 1.013) * 1e5
    TN2   = v[:, 76].astype(float) + 273.15
    Fl    = v[:, 7].astype(float) / RHO_LIQ / 3600.0   # FT103 [kg/hr]->[m3/s]
    Fg    = (FCO2 * (PCO2 / P_bot) * (Tvap / TCO2)
             + FN2 * (PN2 / P_bot) * (Tvap / TN2)) / 3600.0   # [m3/s]

    # inlet CO2 분율: label==6 측정값을 다음 cycle까지 유지 (piecewise-constant hold)
    label = v[:, -1].astype(float)
    origin = v[:, 3].astype(float) / 100.0
    prev = origin[label == 6][0] if np.any(label == 6) else 0.0
    origin_ca = origin.copy()
    for i in range(n):
        if label[i] != 6:
            origin_ca[i] = prev
        else:
            prev = origin_ca[i]
    C_g_in = origin_ca / Vm    # 분율 -> mol/m3

    return dict(n=n, Fg=Fg, Fl=Fl, C_g_in=C_g_in, T=Tliq,
                P_bot=P_bot, P_top=P_top, origin_ca=origin_ca, Vm=Vm,
                label=label,        # 각 시점 측정된 sampling point(1~6) — 결측라벨 mask 용
                co2_meas=origin)    # 각 시점 raw AT400 CO2 분율(비-forwardfill) — obs 라벨용


def build_inputs(prep):
    """샘플 index k(ZOH) -> inputs_fn(t) 반환. hold(다음 값까지 유지)."""
    grid = INI_TIME + DT_IN * np.arange(prep['n'])

    def inputs_fn(t):
        k = int(np.searchsorted(grid, t, side='right') - 1)
        k = max(0, min(k, prep['n'] - 1))
        return prep['Fg'][k], prep['Fl'][k], prep['C_g_in'][k], prep['T'][k]

    return inputs_fn, grid


# ==========================================================================
# 3) 적분 (plumbing 제공) — rhs 구현되면 동작
# ==========================================================================
def simulate(prep):
    inputs_fn, grid = build_inputs(prep)
    n = prep['n']
    t_end = INI_TIME + DT_IN * n

    # 초기상태: 기체=0, 액체=0 (lean), 필요시 조정
    x0 = np.zeros(2 * N_STAGE)

    breaks = np.concatenate(([0.0], grid, [t_end]))
    CG = np.zeros((n, N_STAGE))
    X = x0.copy()
    for s in range(len(breaks) - 1):
        t0, t1 = breaks[s], breaks[s + 1]
        if t1 <= t0:
            continue
        sol = solve_ivp(rhs, (t0, t1), X, args=(inputs_fn,),
                        method='LSODA', rtol=1e-6, atol=1e-9, max_step=DT_IN)
        X = sol.y[:, -1]
        j = np.where(np.isclose(grid, t1))[0]
        if j.size:
            CG[int(j[0])] = X[:N_STAGE]
    # 출력 프로파일(분율): C_g,j * Vm, + point6 = 측정 inlet
    out = np.zeros((n, 6))
    for kk in range(n):
        out[kk, :5] = CG[kk] * prep['Vm'][kk]
        out[kk, 5] = prep['origin_ca'][kk]
    return out, CG


# ==========================================================================
# 4) 검증 하네스 (plumbing 제공) — week1-2.md A.2 통과 기준 5개
# ==========================================================================
def verify(prep, out, CG, ref_csv=None):
    print("  [1] 비음수성        :", end=" ")
    print("PASS" if np.nanmin(out) >= -1e-9 else f"FAIL (min={np.nanmin(out):.3e})")

    # 단조성: 하단(stage5)->탑정(stage1) 로 C_g 감소해야 함 (CG[:,4] >= ... >= CG[:,0])
    mono = np.mean([np.all(np.diff(CG[i][::-1]) <= 1e-9) for i in range(CG.shape[0])])
    print(f"  [2] 프로파일 단조성 : {mono*100:.0f}% 시점에서 상단으로 감소",
          "PASS" if mono > 0.9 else "CHECK")

    # 질량수지(근사): 기체가 잃은 CO2 ≈ 액체가 얻은 CO2 (정성 확인)
    print("  [3] 질량수지 폐합   : (구현 후 in-out-reaction 잔차<1e-3 규모 확인)")

    if ref_csv is not None and os.path.exists(ref_csv):
        ref = pd.read_csv(ref_csv, header=None).values
        m = min(ref.shape[0], out.shape[0])
        r = np.sqrt(((out[:m, :6] - ref[:m, :6]) ** 2).mean())
        print(f"  [*] kinetic_model.py 참조 대비 RMSE(분율) = {r:.3e}")
    print("  [4] mechanistic RMSE: reproduce.py 로 게이트 (아래 안내)")
    print("  [5] inlet staircase : Fig.11 상단 계단거동 정성 확인")


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    data_paths = sorted(glob.glob(os.path.join(REPO, 'data', 'withLabel', '1*.xlsx')))
    out_dir = os.path.join(REPO, 'mechanistic_out')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 74)
    print(" [Week-1] 등온 10상태 forward 시뮬레이션 검증")
    print("=" * 74)
    try:
        for p in data_paths:
            name = os.path.splitext(os.path.basename(p))[0]
            prep = load_inputs(p)
            out, CG = simulate(prep)
            np.savetxt(os.path.join(out_dir, name + '.csv'), out, delimiter=',')
            ref = os.path.join(REPO, 'kinetic_model_py', 'csv_py', name + '.csv')
            print(f"\n[{name}]  rows={out.shape[0]}")
            verify(prep, out, CG, ref_csv=ref)
        print("\n" + "=" * 74)
        print(f" 프로파일 저장: {out_dir}")
        print(" RMSE 게이트:  $env:REPRO_KINETIC_DIR='mechanistic_out'; python reproduce.py")
        print("=" * 74)
    except NotImplementedError as e:
        print("\n[!] 물리 코어가 아직 비어 있습니다 (정상 — 직접 구현 대상):")
        print("    ->", e)
        print("    reaction_rates() 와 rhs() 를 구현한 뒤 다시 실행하세요.")
        print("    가이드는 각 함수의 docstring(논문 Eq.10-13) 참고.")


if __name__ == "__main__":
    main()
