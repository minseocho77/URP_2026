# -*- coding: utf-8 -*-
"""
==============================================================================
 reproduce.py  --  논문 완전 재현 스크립트
------------------------------------------------------------------------------
 논문: Zhuang et al. (2022), "A hybrid data-driven and mechanistic model soft
       sensor for estimating CO2 concentrations for a carbon capture pilot
       plant", Computers in Industry 143, 103747.

 원본 코드(Estimate_CO2_profile.ipynb)의 로직을 .py 로 변환한 것.
 2022년 코드(Keras2/pandas1) -> 2026년 라이브러리(Keras3/pandas3/numpy2)
 호환을 위해 아래 [PATCH] 표시 부분을 수정하였다.

 실행:
   python reproduce.py                 # 기본: Set-3, DAE-16, 재학습(full)
   set REPRO_MODE=DAE & set REPRO_DIM=16 & set REPRO_TEST=3 & set REPRO_EPOCHS=500 & python reproduce.py

 환경변수
   REPRO_MODE   : 차원축소 방식 {DAE, POD, PCA}      (기본 DAE)
   REPRO_DIM    : 축소 차원 {16, 32}                 (기본 16)
   REPRO_TEST   : 테스트 시나리오 {1, 2, 3}          (기본 3)
   REPRO_CALLBACK : 입력시퀀스 과거 길이(=Nseq-1)    (기본 17 -> 18 records)
   REPRO_EPOCHS_DAE / REPRO_EPOCHS_LSTM : 학습 epoch (기본 500/500)
==============================================================================
"""
import os
import glob
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # [PATCH] 창 없이 그림 저장만
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import keras
from keras.layers import Input, Dense, LSTM
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

# ----------------------------------------------------------------------------
# 실행 위치를 스크립트 폴더(레포 루트)로 고정 -> data/, kinetic_model/, results/
# 상대경로가 항상 맞도록.
# ----------------------------------------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 하이퍼파라미터 (환경변수로 오버라이드 가능)
MODE        = os.environ.get("REPRO_MODE", "DAE")
DIM         = int(os.environ.get("REPRO_DIM", "16"))
TEST_SCEN   = os.environ.get("REPRO_TEST", "3")      # '1','2','3'
CALLBACK    = int(os.environ.get("REPRO_CALLBACK", "17"))
EPOCHS_DAE  = int(os.environ.get("REPRO_EPOCHS_DAE", "500"))
EPOCHS_LSTM = int(os.environ.get("REPRO_EPOCHS_LSTM", "500"))

# 테스트 시나리오 -> data_path_list 인덱스 (파일명 오름차순 정렬 기준)
#   idx0=140120_1 idx1=140206_1(Set-2) idx2=140207_1(Set-1) idx3=140207_2
#   idx4=140214_1 idx5=140214_2 idx6=140227_1 idx7=140313_1
TEST_INDEX_MAP = {"1": [2], "2": [1], "3": [1, 2]}
TEST_DF_INDEX_LIST = TEST_INDEX_MAP[TEST_SCEN]


def reset_random_seeds():
    """재현성을 위한 시드 고정 (원본 그대로)."""
    os.environ['PYTHONHASHSEED'] = str(0)
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)


def rmse(pred, label):
    """RMSE = sqrt(mean((pred-label)^2)). 논문 Sec.3 의 평가지표."""
    return np.sqrt(((pred - label) ** 2).mean())


# ==========================================================================
# 1) 전처리 (논문 Sec.2.1 ~ 2.3)
# ==========================================================================
def avgOutPoint1(df):
    """
    [논문 Sec.2.1 / Fig.4b] 샘플링 위치 1(칼럼 최상단)의 CO2 측정값은
    이전 사이클의 고농도 가스가 챔버에 남아 비정상적으로 높다.
    99.5% 이상이 이미 흡수되었으므로, 위치 1의 값을 '위치 2 평균값'으로 대체.
    """
    label_list = list(df['label'])
    conc_list = list(df['AT400(CO2 %)'])
    p = 0; num = 0; i = 0
    sampling2_avgcon = list()
    while i < len(label_list):
        if label_list[i] == 2:
            p = p + conc_list[i]; num = num + 1; i = i + 1
        else:
            if p == 0 and num == 0:
                i = i + 1
            else:
                sampling2_avgcon.append(p / num); i = i + 1; num = 0; p = 0
    i = 0; k = -1
    while i < (len(label_list) - 1):
        if label_list[i] == 1:
            conc_list[i] = sampling2_avgcon[k]; i = i + 1
        else:
            if label_list[i + 1] == 1:
                k = k + 1; i = i + 1
            else:
                i = i + 1
    df['AT400(CO2 %)'] = pd.Series(data=conc_list, index=df.index)
    return df


def columnSeparator(df):
    """
    [논문 Sec.2.1] 가스분석기는 한 번에 한 위치만 측정하므로 나머지 위치는 결측.
    각 샘플링 위치(1~6)별 열을 만들고 '선형 보간'으로 결측값을 채운다.
    (Fig.4b) 큰 변동에서 spurious interpolation 을 피하려 linear 사용.
    """
    for i in range(1, 7):
        new_con = list(); j = 0
        while j < df.shape[0]:
            if df.iloc[j]['label'] == i:
                new_con.append(df.iloc[j]['AT400(CO2 %)']); j = j + 1
            else:
                new_con.append(np.nan); j = j + 1
        df[str(i) + "_sampling"] = pd.Series(data=new_con, index=df.index)
        df[str(i) + "_sampling"] = df[str(i) + "_sampling"].interpolate(method="linear")
    return df


# ==========================================================================
# 2) 차원 축소 (논문 Sec.2.4)
# ==========================================================================
def getU(ens, reduced_dimension):
    """
    [논문 Eq.(3)] POD: X = U S V^T (SVD).  절단 파라미터 d 만큼 U 를 자른다.
    ens: [sample_size, n_features] -> ens.T = X (n_features x m)
    """
    u, s, vh = np.linalg.svd(ens.T, full_matrices=False)
    truncation_parameter = reduced_dimension
    print('Truncation parameter: ', truncation_parameter)
    u1 = u[:, :truncation_parameter]
    return u1


def dfPOD(train_df_list, test_df_list, _train_feature_list, _label_list, reduced_dimension):
    """[논문 Eq.(4)] POD 인코딩: X_d = U_d^T X  (모드 d개로 투영)."""
    print('Train df(1st one) shape before POD: ', train_df_list[0].shape)
    label_list = _label_list.copy()
    train_feature_list = _train_feature_list.copy()

    for i in range(len(train_df_list)):
        if i == 0:
            train_full_values = train_df_list[i][train_feature_list].values
        else:
            train_full_values = np.concatenate((train_full_values, train_df_list[i][train_feature_list].values), axis=0)
    for i in range(len(test_df_list)):
        if i == 0:
            test_full_values = test_df_list[i][train_feature_list].values
        else:
            test_full_values = np.concatenate((test_full_values, test_df_list[i][train_feature_list].values), axis=0)

    n_set = train_full_values.shape[0]
    train_index = list(set(range(0, n_set, 1)) - set(range(0, n_set, 5)))
    test_index = list(set(range(0, n_set, 5)))
    x_train = train_full_values[train_index]; x_val = train_full_values[test_index]
    x_test = test_full_values

    u1 = getU(x_train, reduced_dimension)                 # Eq.(3)
    # 인코딩/디코딩 MSE 로 정보손실 확인
    x_train_decoded = (u1 @ (u1.T @ x_train.T)).T
    x_val_decoded = (u1 @ (u1.T @ x_val.T)).T
    x_test_decoded = (u1 @ (u1.T @ x_test.T)).T
    print("Train: ", mean_squared_error(x_train, x_train_decoded))
    print("Val:   ", mean_squared_error(x_val, x_val_decoded))
    print("Test:  ", mean_squared_error(x_test, x_test_decoded))

    encoded_train = (u1.T @ train_full_values.T).T        # Eq.(4)
    encoded_test = (u1.T @ test_full_values.T).T
    encoded_name_list = ['encoded_{}'.format(i) for i in range(1, encoded_train.shape[1] + 1)]
    train_df_list = _rebuild_df_list(train_df_list, encoded_train, encoded_name_list, label_list)
    test_df_list = _rebuild_df_list(test_df_list, encoded_test, encoded_name_list, label_list)
    return train_df_list, test_df_list


def dfAE(train_df_list, test_df_list, _train_feature_list, _label_list, reduced_dimension):
    """
    [논문 Sec.2.4.2 / Table1 / Eq.(5)-(8)] Denoising AutoEncoder.
      인코더: Dense(64,ReLU) -> Dense(N2,ReLU)     (Eq.5)
      디코더: Dense(64,Linear) -> Dense(90,Sigmoid) (Eq.6)
      노이즈: x_noise ~ N(x, 0.1*Var(X_train))       (Eq.7)
      손실  : MSE(decoded, x)                        (Eq.8)
    """
    print('Train df(1st one) shape before AE: ', train_df_list[0].shape)
    label_list = _label_list.copy()
    train_feature_list = _train_feature_list.copy()

    for i in range(len(train_df_list)):
        if i == 0:
            train_full_values = train_df_list[i][train_feature_list].values
        else:
            train_full_values = np.concatenate((train_full_values, train_df_list[i][train_feature_list].values), axis=0)
    for i in range(len(test_df_list)):
        if i == 0:
            test_full_values = test_df_list[i][train_feature_list].values
        else:
            test_full_values = np.concatenate((test_full_values, test_df_list[i][train_feature_list].values), axis=0)

    n_set = train_full_values.shape[0]
    train_index = list(set(range(0, n_set, 1)) - set(range(0, n_set, 5)))
    test_index = list(set(range(0, n_set, 5)))
    x_train = train_full_values[train_index]; x_val = train_full_values[test_index]
    x_test = test_full_values
    print('AE x_train shape: ', x_train.shape[0])

    # [Eq.(7)] 입력에 분산비례 가우시안 노이즈 추가 (denoising)
    noise_factor = 0.1
    scale_arr = np.var(x_train, axis=0)
    x_train = x_train + noise_factor * np.random.normal(loc=0.0, scale=scale_arr, size=x_train.shape)
    x_test = x_test + noise_factor * np.random.normal(loc=0.0, scale=scale_arr, size=x_test.shape)
    x_train = np.clip(x_train, 0., 1.)
    x_test = np.clip(x_test, 0., 1.)

    # [Eq.(5),(6) / Table1] 5-layer 대칭 AE
    input_img = Input(shape=(x_train.shape[1],))          # [PATCH] Keras3: shape 는 튜플이어야 함
    x = Dense(64, activation='relu', kernel_initializer=ReLu)(input_img)
    encoded = Dense(reduced_dimension, activation='relu', kernel_initializer=ReLu)(x)

    decoder_input = Input(shape=(reduced_dimension,))     # [PATCH] Keras3
    x = Dense(64)(decoder_input)
    decoded = Dense(x_train.shape[1], activation='sigmoid')(x)

    encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
    decoder = Model(inputs=decoder_input, outputs=decoded, name='decoder')
    autoencoder_outputs = decoder(encoder(input_img))
    autoencoder = Model(input_img, autoencoder_outputs, name='autoencoder')
    autoencoder.summary()

    opt = keras.optimizers.Adam(learning_rate=0.001)      # [논문 Sec.3] DAE lr=1e-3
    autoencoder.compile(optimizer=opt, loss="mse")        # [Eq.(8)] MSE 손실

    early = [EarlyStopping(monitor='val_loss', patience=30, verbose=0, restore_best_weights=True)]
    history = autoencoder.fit(x_train, x_train, epochs=EPOCHS_DAE, batch_size=40, shuffle=True,
                              validation_data=(x_val, x_val), callbacks=early)
    # [PATCH] Keras3 는 디렉토리 SavedModel 저장을 model.save 로 지원안함 -> .keras 사용
    autoencoder.save('DAE_repro.keras')

    print('Train RMSE: ', rmse(x_train, autoencoder.predict(x_train, verbose=0)))
    print('Val   RMSE: ', rmse(x_val, autoencoder.predict(x_val, verbose=0)))
    print('Test  RMSE: ', rmse(x_test, autoencoder.predict(x_test, verbose=0)))

    encoder = autoencoder.get_layer("encoder")
    encoded_train = encoder.predict(train_full_values, verbose=0)
    encoded_test = encoder.predict(test_full_values, verbose=0)
    encoded_name_list = ['encoded_{}'.format(i) for i in range(1, encoded_train.shape[1] + 1)]
    train_df_list = _rebuild_df_list(train_df_list, encoded_train, encoded_name_list, label_list)
    test_df_list = _rebuild_df_list(test_df_list, encoded_test, encoded_name_list, label_list)
    return train_df_list, test_df_list


def _rebuild_df_list(df_list, encoded_all, encoded_name_list, label_list):
    """축소된 feature + 원래 label 열로 각 데이터셋 DataFrame 재구성."""
    upper = 0; bottom = 0
    out = []
    for i in range(len(df_list)):
        upper = bottom
        bottom = upper + df_list[i].shape[0]
        out.append(pd.DataFrame(
            np.concatenate((encoded_all[upper:bottom, :], df_list[i][label_list].values), axis=1),
            columns=encoded_name_list + label_list))
    return out


# ==========================================================================
# 3) 시퀀스 데이터셋 생성 (논문 Sec.2.6 / Fig.6)
# ==========================================================================
def getSampleSet(df_list, callback, train_feature_list, label_list):
    """
    [논문 Fig.6] 크기 (callback+1) 의 이동창(moving window)으로 입력 시퀀스 생성.
    입력 feature 에 one-hot 인코딩(6개 샘플링 위치)을 concat -> R^96 (원본 기준).
    라벨은 현재 시점의 6개 위치 CO2 농도 프로파일.
    """
    conc_label_list = []
    for i in range(len(df_list)):
        nset = df_list[i].shape[0] - callback
        conc_input = np.zeros((df_list[i].shape[0], 6))       # one-hot (Sec.2.3)
        # [PATCH] pandas3 의 .values 는 읽기전용 배열을 반환 -> in-place 대입 불가.
        #         쓰기 가능한 float 복사본으로 변환 (원본 로직 동일).
        conc_set = np.array(df_list[i][label_list[1:]].values, dtype=float, copy=True)  # 6개 sampling 열
        conc_set[np.isnan(conc_set)] = 0
        parameter_set = np.array(df_list[i][train_feature_list].values, dtype=float, copy=True)
        for k in range(conc_input.shape[0]):
            conc_input[k][int(df_list[i]['label'][k]) - 1] = 1
        conc_input[np.isnan(conc_input)] = 0
        parameter_set = np.hstack((parameter_set, conc_input))
        sample_set = np.zeros((nset, callback + 1, parameter_set.shape[1]))
        label_set = np.zeros((nset, 1, conc_set.shape[1]))
        for j in range(nset):
            sample_set[j] = parameter_set[0 + j:callback + j + 1]  # 이동창
            label_set[j] = conc_set[callback + j]
        if i == 0:
            total_sample_set = np.zeros((0, callback + 1, parameter_set.shape[1]))
            total_label_set = np.zeros((0, 1, conc_set.shape[1]))
        conc_label_list.append(label_set)
        total_sample_set = np.vstack([total_sample_set, sample_set])
        total_label_set = np.vstack([total_label_set, label_set])
    return total_sample_set, total_label_set, conc_label_list


def getDataSet(path_list, df_list, callback, test_df_index_list, mode, reduced_dimension):
    """전체 전처리 파이프라인: 정규화(Eq.1)->차원축소->시퀀스화."""
    print('Timestamp in selected Test df list: ')
    for i in range(len(test_df_index_list)):
        print(' ', os.path.basename(path_list[test_df_index_list[i]]))

    for i in range(len(df_list)):
        df_list[i] = df_list[i].set_index('time')
        df_list[i] = avgOutPoint1(df_list[i])              # Sec.2.1

    # Train/Test 분리
    test_df_list = list(pd.Series(df_list)[test_df_index_list])
    train_df_list = list(pd.Series(df_list)[list(set(range(0, len(df_list), 1)) - set(test_df_index_list))])

    # [Eq.(1)] Min-Max 정규화 (train 으로만 fit)
    for i in range(len(train_df_list)):
        if i == 0:
            tmp_full_values = train_df_list[i].values
            tmp_conc_values = train_df_list[i]['AT400(CO2 %)'].values
        else:
            tmp_full_values = np.concatenate((tmp_full_values, train_df_list[i].values), axis=0)
            tmp_conc_values = np.concatenate((tmp_conc_values, train_df_list[i]['AT400(CO2 %)'].values), axis=0)
    tmp_full_values = tmp_full_values[:, :-1]              # label 열 제외
    general_scaler = MinMaxScaler()
    conc_scaler = MinMaxScaler()
    general_scaler.fit(tmp_full_values)
    conc_scaler.fit(tmp_conc_values.reshape(-1, 1))
    print('Conc scaler max: ', conc_scaler.data_max_)

    for i in range(len(df_list)):
        # [PATCH] pandas3: 슬라이스 직접대입 대신 .iloc[:, :-1] 위치기반 대입 (dtype 안전)
        scaled = general_scaler.transform(df_list[i].iloc[:, :-1].values)
        df_list[i].iloc[:, :-1] = scaled
        df_list[i] = columnSeparator(df_list[i])           # 선형보간
        df_list[i] = df_list[i].fillna(0)

    # Train/Test 재분리 (보간 후)
    test_df_list = list(pd.Series(df_list)[test_df_index_list])
    train_df_list = list(pd.Series(df_list)[list(set(range(0, len(df_list), 1)) - set(test_df_index_list))])

    label_list = ['label', '1_sampling', '2_sampling', '3_sampling', '4_sampling', '5_sampling', '6_sampling']
    train_feature_list = list(np.sort(list(set(df_list[0].columns) - set(label_list))))

    if mode == 'PCA':
        # [논문 Sec.2.4.1] PCA 는 주성분 기여도로 원본 feature 를 '선택'
        if reduced_dimension == 16:
            train_feature_list = ['TT113(0C)', 'TT300(0C)', 'TT302(0C)', 'TT202(0C)', 'TT304(0C)', "TT400(0C)",
                                  "TT410(0C)", 'PT103(barg)', 'PT402(barg)', 'PT403(barg)', 'PT102(barg)', 'AT300(pH)',
                                  'FT302(kg/hr)', 'FT303m3/hr', 'FT304(kg/hr)', 'FT103(kg/hr)']
        else:
            train_feature_list = ['TT302(0C)', 'TT300(0C)', 'TT401(0C)', 'TT303(0C)', 'TT400(0C)', 'TT113(0C)',
                                  'TT301(0C)', 'TT202(0C)', 'TT412(0C)', 'TT309(0C)', 'TT110a(0C)', 'TT410(0C)',
                                  'TT112(0C)', 'PT103(barg)', 'PT402(barg)', 'PT401(barg)', 'TT404(0C)', 'TT214(0C)',
                                  'FT304(kg/hr)', 'AT100(pH)', 'FT303m3/hr', 'AT300(pH)', 'FT105(L/min)', 'FT103(kg/hr)',
                                  'PT403(barg)', 'TT304(0C)', 'FT301m3/hr', 'TT107(0C)', 'PT110(barg)', 'PT111(barg)',
                                  'TT210(0C)', 'TT211(0C)']
    if mode == 'DAE':
        train_df_list, test_df_list = dfAE(train_df_list, test_df_list, train_feature_list, label_list, reduced_dimension)
        train_feature_list = list(np.sort(list(set(train_df_list[0].columns) - set(label_list))))
    if mode == 'POD':
        train_df_list, test_df_list = dfPOD(train_df_list, test_df_list, train_feature_list, label_list, reduced_dimension)
        train_feature_list = list(np.sort(list(set(train_df_list[0].columns) - set(label_list))))

    train_sample_set, train_label_set, _ = getSampleSet(train_df_list, callback, train_feature_list, label_list)
    test_sample_set, test_label_set, test_conc_list = getSampleSet(test_df_list, callback, train_feature_list, label_list)
    return (train_sample_set, train_label_set, test_sample_set, test_label_set,
            conc_scaler, train_df_list, test_df_list, test_conc_list)


# ==========================================================================
# 4) 공분산 국소화 (논문 Eq.(19), Gaspari-Cohn)
# ==========================================================================
def GCfunc(i, j, corr_length):
    """[논문 Eq.(19)] Gaspari-Cohn 국소화 함수. r=|i-j|/L."""
    r = abs(i - j) / corr_length
    if 0 <= r < 1:
        return (1 - pow(r, 2) * 5 / 3 + pow(r, 3) * 5 / 8 + pow(r, 4) * 0.5 - pow(r, 5) * 0.25)
    if 1 <= r < 2:
        return (4 - 5 * r + pow(r, 2) * 5 / 3 + pow(r, 3) * 5 / 8 - pow(r, 4) * 0.5 + pow(r, 5) / 12 - 2 / 3 / r)
    return 0


def covLoc(B, corr_length):
    """[논문 Eq.(19) 아래] B_ij <- B_ij * G(rho) 로 공분산 국소화."""
    ls = B.shape[0]
    for i in range(ls):
        for j in range(ls):
            B[i][j] = B[i][j] * GCfunc(i, j, corr_length)
    return B


# ==========================================================================
# 5) LSTM 데이터 기반 모델 (논문 Sec.2.6 / Table3)
# ==========================================================================
def trainSequenceLSTM(state, train_sample_set, train_label_set, test_sample_set, test_label_set, conc_scaler):
    n_sequence = train_sample_set.shape[1]
    n_feature = train_sample_set.shape[2]
    n_set = train_sample_set.shape[0]
    print('train_sample_set shape: ', train_sample_set.shape)

    # [Fig.6] 검증셋 = 5개마다 1개 (20%)
    train_index = list(set(range(0, n_set, 1)) - set(range(0, n_set, 5)))
    test_index = list(set(range(0, n_set, 5)))
    x_train = train_sample_set[train_index]; x_val = train_sample_set[test_index]
    y_train = train_label_set[train_index]; y_val = train_label_set[test_index]
    x_test = test_sample_set; y_test = test_label_set
    print('Train len: %d, Val len: %d' % (x_train.shape[0], x_val.shape[0]))

    if state == 'train':
        # [Table3] LSTM(100) -> Dropout(0.1) -> Dense(100) -> Dropout -> Dense(100)
        #          -> Dropout -> Dense(6, sigmoid).  출력평균은 numpy 에서 처리.
        model = Sequential()
        model.add(Input(shape=(n_sequence, n_feature)))    # [PATCH] Keras3 명시적 Input
        model.add(LSTM(100, return_sequences=True))
        model.add(keras.layers.Dropout(0.1))
        model.add(Dense(100))
        model.add(keras.layers.Dropout(0.1))
        model.add(Dense(100))
        model.add(keras.layers.Dropout(0.1))
        model.add(Dense(6, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=0.00001)  # [논문 Sec.3] LSTM lr=1e-5
        model.compile(optimizer=opt, loss='mse')
        model.summary()
        early = [EarlyStopping(monitor='val_loss', patience=70, verbose=0, restore_best_weights=True)]
        train_log = model.fit(x_train, y_train, batch_size=40, epochs=EPOCHS_LSTM, shuffle=True,
                              validation_data=(x_val, y_val), callbacks=early)
        model.save('LSTM_MSE_repro.keras')                  # [PATCH] .keras 포맷
        plt.figure()
        plt.plot(train_log.history['loss']); plt.plot(train_log.history['val_loss'])
        plt.title('model loss'); plt.ylabel('loss'); plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('lstm_loss_repro.png', dpi=120); plt.close()
    else:
        model = keras.models.load_model('LSTM_MSE_repro.keras')

    # 테스트/검증 평가 (원 스케일로 역변환 후 RMSE)
    predict_raw = model.predict(x_test, verbose=0)          # raw 출력 (nset, Nseq, 6)
    # [PATCH] 원본 노트북은 fusion 에 저장된 npy(=시퀀스평균+역정규화, (nset,6))를 사용.
    #  reproduce 에서는 여기서 동일하게 시퀀스 평균 -> 원 농도(%) 스케일로 변환해 fusion 에 전달.
    #  (논문 Table3 의 'Mean' 계층 = np.mean(axis=1), Eq.(1) 역변환)
    predict_set = conc_scaler.inverse_transform(np.mean(predict_raw, axis=1))  # (nset, 6), % 스케일
    test_rmse = rmse(predict_set,
                     conc_scaler.inverse_transform(y_test.reshape(y_test.shape[0], 6)))
    print('LSTM Test RMSE: ', test_rmse)

    val_output_set = model.predict(x_val, verbose=0)
    val_rmse = rmse(conc_scaler.inverse_transform(np.mean(val_output_set, axis=1)),
                    conc_scaler.inverse_transform(y_val.reshape(y_val.shape[0], 6)))
    print('LSTM Val  RMSE: ', val_rmse)

    # [Eq.(18)] R = Cov(y_LSTM - y_linear)  (검증셋으로 데이터모델 불확실성 추정)
    R = np.cov((conc_scaler.inverse_transform(np.mean(val_output_set, axis=1)) -
                conc_scaler.inverse_transform(y_val.reshape(y_val.shape[0], 6))).T)
    R = covLoc(R, 2)                                        # Eq.(19) 국소화
    np.save('R_repro.npy', R)
    return model, predict_set, test_rmse, val_rmse


# ==========================================================================
# 6) 출력 융합 (논문 Sec.2.8 / Eq.(17))
# ==========================================================================
def VAR_3D(xb, Y, H, B, R):
    """
    [논문 Eq.(17)] 3D-Var / Kalman analysis 형태의 융합.
      칼만게인 K = B H^T (H B H^T + R)^-1
      사후추정 xa = xb + K (Y - H xb)
    H=I 일 때 Eq.(17) y = y_mec + B(R+B)^-1 (y_LSTM - y_mec) 와 등가.
      xb=y_mec(메커니즘, 사전추정), Y=y_LSTM(데이터모델, 관측 대용)
    """
    dim_x = xb.size
    Y = Y.reshape(Y.size, 1)
    xb1 = np.copy(xb).reshape(xb.size, 1)
    K = np.dot(B, np.dot(np.transpose(H), np.linalg.pinv(np.dot(H, np.dot(B, np.transpose(H))) + R)))
    xa = np.copy(xb1 + np.dot(K, (Y - np.dot(H, xb1))))
    return xa.ravel()


def run_fusion(data_path_list, callback, test_df_index_list, conc_scaler,
               train_label_set, test_label_set, predict_set):
    """메커니즘 CSV 로드 -> B 계산(Eq.18) -> 테스트셋 융합(Eq.17) -> RMSE.

    [PATCH] REPRO_KINETIC_DIR 환경변수로 메커니즘 프로파일 출처를 선택.
      - 기본 'kinetic_model'        : 원저자 MATLAB/Simulink 가 생성한 CSV
      - 'kinetic_model_py/csv_py'   : 우리가 Python 으로 재이식해 재계산한 CSV
    """
    kinetic_dir = os.environ.get("REPRO_KINETIC_DIR", "kinetic_model")
    dynamic_path_list = sorted(glob.glob(os.path.join(kinetic_dir, '1*.csv')))
    print(f"[fusion] 메커니즘 프로파일 출처: {kinetic_dir}  (파일 {len(dynamic_path_list)}개)")
    test_name_list = [os.path.splitext(os.path.basename(data_path_list[k]))[0] for k in test_df_index_list]

    dynamic_train_list = []; dynamic_test_list = []
    for path in dynamic_path_list:
        name = os.path.splitext(os.path.basename(path))[0]
        if name not in test_name_list:
            dynamic_train_list.append(pd.read_csv(path, header=None))
        else:
            dynamic_test_list.append(pd.read_csv(path, header=None))

    # [Eq.(18)] B = Cov(y_mec - y_linear)  (메커니즘 모델 불확실성)
    for i in range(len(dynamic_train_list)):
        if i == 0:
            train_full_values = dynamic_train_list[i].values[callback:, :]
        else:
            train_full_values = np.concatenate((train_full_values, dynamic_train_list[i].values[callback:, :]), axis=0)
    B = np.cov((train_full_values * 100 -
                conc_scaler.inverse_transform(train_label_set.reshape(train_label_set.shape[0], 6))).T)
    B = covLoc(B, 2)
    np.save('B_repro.npy', B)

    for i in range(len(dynamic_test_list)):
        if i == 0:
            test_full_values = dynamic_test_list[i].values[callback:, :]
        else:
            test_full_values = np.concatenate((test_full_values, dynamic_test_list[i].values[callback:, :]), axis=0)

    R = np.load('R_repro.npy')
    mech_rmse = rmse(test_full_values * 100,
                     conc_scaler.inverse_transform(test_label_set.reshape(test_label_set.shape[0], 6)))

    # [Eq.(17)] 융합
    H = np.eye(6)
    conc_set = test_full_values.copy()
    for i in range(conc_set.shape[0]):
        conc_set[i] = VAR_3D(test_full_values[i] * 100, predict_set[i], H, B, R)
    fused_rmse = rmse(conc_set, conc_scaler.inverse_transform(test_label_set.reshape(test_label_set.shape[0], 6)))

    # [PATCH] Fig.11 재현용 배열 저장 (측정/메커니즘/DAE-LSTM/Fused). set 별 파일명.
    tag = f"set{TEST_SCEN}"
    np.save(f'fig11_{tag}_meas.npy',
            conc_scaler.inverse_transform(test_label_set.reshape(test_label_set.shape[0], 6)))
    np.save(f'fig11_{tag}_mech.npy', test_full_values * 100)
    np.save(f'fig11_{tag}_daelstm.npy', predict_set)
    np.save(f'fig11_{tag}_fused.npy', conc_set)
    return mech_rmse, fused_rmse, B, R


# ==========================================================================
# MAIN
# ==========================================================================
def load_all_dataframes(data_path_list):
    dfs = []
    for p in data_path_list:
        xls = pd.ExcelFile(p)
        data_df = pd.read_excel(xls, sheet_name=0, index_col=0, header=[0, 1])
        data_df.columns = data_df.columns.map(''.join)
        data_df = data_df.rename_axis('time').reset_index()
        tmp_name = list(data_df.columns); tmp_name[-1] = 'label'
        data_df.columns = tmp_name
        dfs.append(data_df)
    return dfs


def main():
    global ReLu
    reset_random_seeds()
    ReLu = keras.initializers.HeUniform()

    data_path_list = sorted(glob.glob('data/withLabel/1*.xlsx'))
    print("=" * 70)
    print(f" 설정: MODE={MODE}, DIM={DIM}, TEST=Set-{TEST_SCEN} "
          f"(idx={TEST_DF_INDEX_LIST}), CALLBACK={CALLBACK} "
          f"(={CALLBACK+1} records), DAE_ep={EPOCHS_DAE}, LSTM_ep={EPOCHS_LSTM}")
    print(" 데이터 파일:")
    for i, p in enumerate(data_path_list):
        print(f"   [{i}] {os.path.basename(p)}")
    print("=" * 70)

    list_of_origin_df = load_all_dataframes(data_path_list)
    list_of_processed_df = [d.copy() for d in list_of_origin_df]

    (train_sample_set, train_label_set, test_sample_set, test_label_set,
     conc_scaler, train_df_list, test_df_list, test_conc_list) = getDataSet(
        data_path_list, list_of_processed_df, CALLBACK, TEST_DF_INDEX_LIST, MODE, DIM)

    model, predict_set, lstm_test_rmse, lstm_val_rmse = trainSequenceLSTM(
        'train', train_sample_set, train_label_set, test_sample_set, test_label_set, conc_scaler)

    mech_rmse, fused_rmse, B, R = run_fusion(
        data_path_list, CALLBACK, TEST_DF_INDEX_LIST, conc_scaler,
        train_label_set, test_label_set, predict_set)

    # 논문 대비 결과 요약
    paper = {
        "1": dict(Mechanistic=0.117, DAE_LSTM=0.205, Fused=0.102, SSAE=0.168),
        "2": dict(Mechanistic=0.325, DAE_LSTM=0.281, Fused=0.204, SSAE=0.779),
        "3": dict(Mechanistic=0.190, DAE_LSTM=0.201, Fused=0.123, SSAE=0.412),
    }[TEST_SCEN]

    print("\n" + "=" * 70)
    print(f" 재현 결과 (Set-{TEST_SCEN}, {MODE}-{DIM}, {CALLBACK+1} records)")
    print("=" * 70)
    print(f" {'Model':<14}{'재현(RMSE)':>14}{'논문(RMSE)':>14}")
    print(f" {'-'*42}")
    print(f" {'Mechanistic':<14}{mech_rmse:>14.3f}{paper['Mechanistic']:>14.3f}")
    print(f" {'DAE-LSTM':<14}{lstm_test_rmse:>14.3f}{paper['DAE_LSTM']:>14.3f}")
    print(f" {'Fused':<14}{fused_rmse:>14.3f}{paper['Fused']:>14.3f}")
    print(f" {'SSAE(논문값)':<14}{'-':>14}{paper['SSAE']:>14.3f}")
    print("=" * 70)
    print(" 주: 재학습 결과는 TF/Keras 버전·난수 차이로 논문값과 정확히 일치하지 않을 수 있음.")


if __name__ == "__main__":
    main()
