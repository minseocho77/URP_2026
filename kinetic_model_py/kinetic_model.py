# -*- coding: utf-8 -*-
"""
==============================================================================
 kinetic_model.py  --  MATLAB/Simulink 메커니즘 모델의 Python 물리-충실 이식본
------------------------------------------------------------------------------
 원본:
   kinetic_model/simple_cstr.m         : 단일 CSTR 단(stage)의 S-Function ODE
   kinetic_model/kinetic_model.slx     : 5단을 향류(countercurrent)로 연결한 Simulink
   kinetic_model/results_generation.m  : Excel 입력 전처리 + 시뮬 구동 + CSV 저장

 이 파일은 위 3개를 CSV 재사용 없이 "처음부터" Python 으로 재구현한 것.
 .slx(바이너리)를 ZIP→XML 파싱해 역설계한 위상(topology):

   액체(MEA) 하강  A→B→C→D→E          기체(CO2) 상승  E→D→C→B→A
   ┌───────────────────────────────────────────┐
   │ Stage A (top)  → CA1 = sampling point 1 (탑정, CO2 최저) │
   │ Stage B        → CA2 = point 2                          │
   │ Stage C        → CA3 = point 3                          │
   │ Stage D        → CA4 = point 4                          │
   │ Stage E (bot)  → CA5 = point 5                          │
   └───────────────────────────────────────────┘
   기체 유입(하단) = C_Ain(측정 point6=Origin_CA)     point6 = 측정값(비시뮬)

 반응 (simple_cstr.m 주석): CO2(g) + MEA -> 흡수(bound)
   R1 = k1*Ca*Cb   (흡수, k1 Arrhenius)
   R2 = k2*Cc      (탈착)
==============================================================================
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

os.chdir(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ==========================================================================
# 0) 물리 상수 (simple_cstr.m mdlDerivatives 그대로)
# ==========================================================================
D_COL   = 25.4 / 100.0                       # 칼럼 직경 [m] (25.4/100)  ※ .m 값 사용
VSEC    = 1.872 * 0.25 * np.pi * D_COL**2    # 한 단(section) 체적 [m^3]
VLIQ    = VSEC * 0.035                        # 액상 holdup 체적
VVAP    = VSEC * 0.985 * (1 - 0.035)          # 기상 체적
ROU     = 1004.0                              # 액 밀도 [kg/m3]
CPLIQ   = 3414.5                              # 액 비열 [J/(kg K)]
DH1     = -32000.0                            # 흡수열 [J/mol CO2]
DH2     =  32000.0                            # 탈착열
EA1     = 69050.0                             # 활성화E [J/mol]
EA2     = EA1 + abs(DH1)                      # = 101050
RGAS    = 8.314

# 적합된 반응 파라미터 (Simulink S-Function 블록 Parameters 1번, 5단 공통)
#   Param = [k01, alpha1, beta1, k02, alpha2, beta2]
K01, ALPHA1, BETA1, K02, ALPHA2, BETA2 = 500.6, 1.0, 1005.5, 4380.0, 1.0, 1263.0

N_STAGE = 5
INI_TIME = 600.0     # results_generation.m: 초기 안정화 구간 [s]
DT_IN    = 43.0      # FromWorkspace SampleTime [s]

RHO_LIQ = 1004.0     # 전처리용 액 밀도


# ==========================================================================
# 1) 단일 CSTR 단 미분식  (simple_cstr.m mdlDerivatives, line 138-148)
# ==========================================================================
def stage_deriv(x, u):
    """
    x = [Fliq, Fvap, Ca(CO2g), Cb(MEA), Cc(bound), T]      (상태 6)
    u = [Fliq_in, Fvap_in, Ca_in, Cb_in, Cc_in, T_in, P_bot, P_top] (입력 8)
    반환: dx/dt (6,)
    """
    Fliq, Fvap, Ca, Cb, Cc, T = x
    Fliq_in, Fvap_in, Ca_in, Cb_in, Cc_in, T_in, P_bot, P_top = u

    k1 = K01 * np.exp(-EA1 / RGAS / (ALPHA1 * T + BETA1))
    k2 = K02 * np.exp(-EA2 / RGAS / (ALPHA2 * T + BETA2))
    R1 = k1 * Ca * Cb        # mol/(m3 s)  흡수
    R2 = k2 * Cc             #             탈착
    Vm = RGAS * T / ((P_bot + P_top) / 2.0)   # molar specific volume [m3/mol]

    dx = np.empty(6)
    dx[0] = Fliq_in - Fliq                                        # dFliq/dt
    dx[1] = Fvap_in - Fvap + (R2 - R1) * VLIQ * Vm                # dFvap/dt
    dx[2] = Fvap_in / VVAP * (Ca_in - Ca) + (-R1 + R2) * (VLIQ / VVAP)   # dCa/dt
    dx[3] = Fliq_in / VLIQ * (Cb_in - Cb) - R1 + R2              # dCb/dt
    dx[4] = Fliq_in / VLIQ * (Cc_in - Cc) + R1 - R2             # dCc/dt
    dx[5] = (Fliq_in / VLIQ) * (T_in - 298.0) - (Fliq / VLIQ) * (T - 298.0) \
            + (1.0 / (ROU * CPLIQ)) * (R1 * (-DH1) + R2 * (-DH2))   # dT/dt
    return dx


# ==========================================================================
# 2) 5단 향류 칼럼 = 30상태 연립 ODE
#    (Simulink 위상 역설계 결과: 액체 A→E 하강, 기체 E→A 상승)
# ==========================================================================
def column_deriv(t, X, bc):
    """
    X : (30,) = 5단 × 6상태, stage 0=A(top) .. 4=E(bottom)
    bc: dict of 경계입력 스칼라 (해당 ZOH 구간에서 상수)
        Fliq_top, Cb_top(=2465.619), Cc_top(=0), T_top(=Tliq_in),
        Fvap_bot, Ca_bot(=C_Ain), P_bot, P_top
    """
    Xs = X.reshape(N_STAGE, 6)
    dX = np.empty((N_STAGE, 6))
    P_bot, P_top = bc['P_bot'], bc['P_top']

    for i in range(N_STAGE):
        # --- 액상 입력 (위 단 i-1 에서 하강) ---
        if i == 0:  # Stage A: 신선한 MEA
            Fliq_in, Cb_in, Cc_in, T_in = bc['Fliq_top'], bc['Cb_top'], bc['Cc_top'], bc['T_top']
        else:
            up = Xs[i - 1]
            Fliq_in, Cb_in, Cc_in, T_in = up[0], up[3], up[4], up[5]
        # --- 기상 입력 (아래 단 i+1 에서 상승) ---
        if i == N_STAGE - 1:  # Stage E: 신선한 기체(측정 입구)
            Fvap_in, Ca_in = bc['Fvap_bot'], bc['Ca_bot']
        else:
            dn = Xs[i + 1]
            Fvap_in, Ca_in = dn[1], dn[2]

        u = (Fliq_in, Fvap_in, Ca_in, Cb_in, Cc_in, T_in, P_bot, P_top)
        dX[i] = stage_deriv(Xs[i], u)

    return dX.ravel()


# ==========================================================================
# 3) Excel 전처리  (results_generation.m line 5-92 그대로)
#    MATLAB input_df(:,k) == pandas df.iloc[:,k]  (col 매핑 확인 완료)
# ==========================================================================
def load_and_prepare(path):
    xls = pd.ExcelFile(path)
    df = pd.read_excel(xls, sheet_name=0, index_col=0, header=[0, 1])
    df.columns = df.columns.map(''.join)
    df = df.rename_axis('time').reset_index()
    cols = list(df.columns); cols[-1] = 'label'
    df.columns = cols

    v = df.values                       # [time, sensors..., label]
    n = df.shape[0]

    # --- MATLAB 컬럼 인덱스 (== pandas iloc) ---
    P_bot = (v[:, 31].astype(float) + 1.013) * 1e5    # PT110 [Pa]
    P_top = (v[:, 37].astype(float) + 1.013) * 1e5    # PT403 [Pa]
    avgP  = (P_bot + P_top) / 2.0
    Tliq_in = 0.5 * (v[:, 65].astype(float) + v[:, 66].astype(float)) + 273.15   # TT210,TT211 [K]
    Tvap_in = v[:, 43].astype(float) + 273.15         # TT107 [K]
    Vm      = RGAS * Tvap_in / avgP
    FCO2_in = v[:, 17].astype(float)                  # FT301 [m3/h]
    PCO2_in = (v[:, 36].astype(float) + 1.013) * 1e5  # PT402 [Pa]
    TCO2_in = v[:, 87].astype(float) + 273.15         # TT410 [K]
    FN2_in  = v[:, 19].astype(float)                  # FT303 [m3/h]
    PN2_in  = (v[:, 37].astype(float) + 1.013) * 1e5  # PT403 [Pa]
    TN2_in  = v[:, 76].astype(float) + 273.15         # TT304 [K]
    Fliq_in = v[:, 7].astype(float) / RHO_LIQ / 3600.0  # FT103 [kg/hr]->[m3/s]

    # 기상 총 유량 Fvap [m3/s]  (results_generation.m line 67-68)
    Fvap = FCO2_in * (PCO2_in / P_bot) * (Tvap_in / TCO2_in) \
         + FN2_in * (PN2_in / P_bot) * (Tvap_in / TN2_in)      # m3/h
    Fvap = Fvap / 3600.0                                        # m3/s

    # --- Origin_CA: point6(label==6) 측정 CO2 분율을 forward-fill (line 70-91) ---
    label = v[:, -1].astype(float)
    origin = v[:, 3].astype(float) / 100.0   # AT400(CO2 %) -> 분율
    first_conc = 0.0; found = False
    for i in range(n):
        if label[i] == 6 and not found:
            first_conc = origin[i]; found = True
            break
    origin_ca = origin.copy()
    prev = first_conc
    for i in range(n):
        if label[i] != 6:
            origin_ca[i] = prev
        else:
            prev = origin_ca[i]
    C_Ain = origin_ca / Vm    # 분율 -> mol/m3

    return dict(n=n, P_bot=P_bot, P_top=P_top, Tliq_in=Tliq_in,
                Fliq_in=Fliq_in, Fvap=Fvap, C_Ain=C_Ain, origin_ca=origin_ca)


# ==========================================================================
# 4) 한 실험(파일) 시뮬레이션  -> (n_set, 6) CO2 분율 프로파일
# ==========================================================================
def simulate_file(prep):
    n = prep['n']
    run_time = DT_IN * n
    t_end = INI_TIME + run_time

    # 입력 시계열 시각: t=0(=첫값) + [ini_time, ini_time+43, ...] (n개)
    #   (results_generation.m: timeseries(...,ini_time:43:ini_time+run_time-1) + addsample @0)
    grid = INI_TIME + DT_IN * np.arange(n)      # 각 샘플의 시작시각 [600, 643, ...]

    def bc_at(k):
        """샘플 인덱스 k 의 경계입력 (ZOH)."""
        return dict(
            Fliq_top=prep['Fliq_in'][k], Cb_top=2465.619, Cc_top=0.0,
            T_top=prep['Tliq_in'][k],
            Fvap_bot=prep['Fvap'][k], Ca_bot=prep['C_Ain'][k],
            P_bot=prep['P_bot'][k], P_top=prep['P_top'][k])

    # 초기상태 X_ss = [Fliq_in[0], Fvap[0], 0, 2465.619, 0, Tliq_in[0]] (5단 동일)
    x0_stage = np.array([prep['Fliq_in'][0], prep['Fvap'][0], 0.0,
                         2465.619, 0.0, prep['Tliq_in'][0]])
    X = np.tile(x0_stage, N_STAGE)

    # ZOH 구간 경계: [0, 600, 643, 686, ..., t_end]
    breaks = np.concatenate(([0.0], grid, [t_end]))
    # 구간별 상수입력 인덱스: [0..599]=샘플0, [600..642]=샘플0, [643..]=샘플1 ...
    #   (결과 CA 는 각 grid 시각에서 채취)
    CA = np.zeros((n, 6))

    seg_idx = 0  # 현재 사용하는 입력 샘플 인덱스
    for s in range(len(breaks) - 1):
        t0, t1 = breaks[s], breaks[s + 1]
        if t1 <= t0:
            continue
        # 이 구간에서 유효한 입력 샘플: grid 상에서 t0 이하의 마지막 샘플
        k = int(np.searchsorted(grid, t0, side='right') - 1)
        k = max(0, min(k, n - 1))
        bc = bc_at(k)
        sol = solve_ivp(column_deriv, (t0, t1), X, args=(bc,),
                        method='LSODA', rtol=1e-6, atol=1e-9, max_step=DT_IN)
        X = sol.y[:, -1]
        # grid 시각 t1 에 도달했으면 그 시점 프로파일 기록
        j = np.where(np.isclose(grid, t1))[0]
        if j.size:
            kk = int(j[0])
            Xs = X.reshape(N_STAGE, 6)
            P_bot, P_top = prep['P_bot'][kk], prep['P_top'][kk]
            for st in range(N_STAGE):
                Ca, T = Xs[st, 2], Xs[st, 5]
                Vm = RGAS * T / ((P_bot + P_top) / 2.0)
                CA[kk, st] = Ca * Vm            # CO2 분율 (Ca*Vm)
            CA[kk, 5] = prep['origin_ca'][kk]   # point6 = 측정 입구값
    return CA


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    data_paths = sorted(glob.glob(os.path.join(REPO, 'data', 'withLabel', '1*.xlsx')))
    out_dir = os.path.join(os.path.dirname(__file__), 'csv_py')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 74)
    print(" Python 메커니즘 모델 (5단 향류 CSTR) - 원본 MATLAB CSV 없이 재계산")
    print("=" * 74)

    for p in data_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        prep = load_and_prepare(p)
        CA = simulate_file(prep)
        np.savetxt(os.path.join(out_dir, name + '.csv'), CA, delimiter=',')

        # 원본 MATLAB CSV 와 비교 (검증용, 입력으로는 사용 안 함)
        ref_path = os.path.join(REPO, 'kinetic_model', name + '.csv')
        if os.path.exists(ref_path):
            ref = pd.read_csv(ref_path, header=None).values
            m = min(ref.shape[0], CA.shape[0])
            diff = CA[:m] - ref[:m]
            rmse_frac = np.sqrt((diff ** 2).mean())
            # 지점별 RMSE(%) — fusion 은 *100 스케일
            per_pt = np.sqrt((diff[:, :6] ** 2).mean(axis=0)) * 100
            print(f"\n[{name}]  rows py={CA.shape[0]} ref={ref.shape[0]}")
            print(f"   전체 RMSE(분율) = {rmse_frac:.3e}   (x100 => {rmse_frac*100:.4f} %p)")
            print("   지점별 RMSE(%p): " +
                  ", ".join(f"P{ii+1}={per_pt[ii]:.4f}" for ii in range(6)))
        else:
            print(f"\n[{name}]  (원본 CSV 없음)  rows={CA.shape[0]}")

    print("\n" + "=" * 74)
    print(f" Python 재계산 CSV 저장 위치: {out_dir}")
    print("=" * 74)


if __name__ == "__main__":
    main()
