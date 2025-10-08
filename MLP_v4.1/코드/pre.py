import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------
# 공통 파라미터(학습/추론 모두 동일하게 유지)
# -------------------------------
WAVELET = 'db4'
LEVEL = 1
THRESH_MODE = 'soft'   # 'soft' 권장
WINDOW_SIZE = 16      # 최소 2**LEVEL 이상 권장 (db4면 128~512 추천)
HOP_SIZE = 1           # 1이면 매 샘플 출력(최대 지연 WINDOW_SIZE)
BORDER_MODE = 'symmetric'  # pywt.pad 기본과 일치; 양쪽 패딩

mag_cols = ['Mag_X', 'Mag_Y', 'Mag_Z']
ori_cols = ['Ori_X', 'Ori_Y', 'Ori_Z']
all_sensor_cols = mag_cols + ori_cols

# -------------------------------
# 웨이브렛 유틸
# -------------------------------
def wavelet_denoise_window(x, wavelet=WAVELET, level=LEVEL, mode=THRESH_MODE, uthresh=None):
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode=BORDER_MODE)
    # sigma 추정 (최상위 detail 계수)
    if uthresh is None:
        d = coeffs[-1]
        if len(d) == 0:
            return x.copy()
        sigma = np.median(np.abs(d)) / 0.6745
        if sigma == 0:
            return x.copy()
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    # A(approx) 제외, D(detail)만 임계값
    den = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]]
    y = pywt.waverec(den, wavelet=wavelet, mode=BORDER_MODE)
    return y[:len(x)]  # 길이 보정

def overlap_add(windows, hop_size=HOP_SIZE, total_len=None):
    """ 윈도우 배열(list of np.array)을 hop으로 이어붙이며 평균 """
    win_len = len(windows[0])
    if total_len is None:
        total_len = (len(windows)-1)*hop_size + win_len
    out = np.zeros(total_len, dtype=float)
    wts = np.zeros(total_len, dtype=float)

    for i, w in enumerate(windows):
        start = i*hop_size
        end = start + win_len
        out[start:end] += w
        wts[start:end] += 1.0
    wts[wts == 0] = 1.0
    return out / wts

# -------------------------------
# B안(fixed)을 위한 전역 sigma 추정 (학습 단계에서 1회)
# -------------------------------
def estimate_global_sigma(series, wavelet=WAVELET, level=LEVEL):
    """ 단일 축 시계열에서 level 최상위 detail 계수의 MAD 기반 sigma """
    coeffs = pywt.wavedec(series, wavelet=wavelet, level=level, mode=BORDER_MODE)
    d = coeffs[-1]
    if len(d) == 0:
        return 0.0
    return np.median(np.abs(d)) / 0.6745

# -------------------------------
# 0) 데이터 로드
# -------------------------------
df = pd.read_csv('merged.csv')
print("✅ 파일 불러오기 완료!", df.shape)

# -------------------------------
# 1) Hard-Iron
# -------------------------------
bias = {}
for col in mag_cols:
    max_val = df[col].max()
    min_val = df[col].min()
    b = (max_val + min_val) / 2.0
    bias[col] = float(b)
    df[col] = df[col] - b
print("✅ Hard-Iron 완료")

# -------------------------------
# 2) Soft-Iron (선형 보정)
# -------------------------------
cov = np.cov(df[mag_cols].values.T)
eig_vals, eig_vecs = np.linalg.eigh(cov)
eps = 1e-12
scale = np.diag(1.0 / np.sqrt(np.clip(eig_vals, eps, None)))
soft_iron_matrix = eig_vecs @ scale @ eig_vecs.T
df[mag_cols] = (soft_iron_matrix @ df[mag_cols].values.T).T
print("✅ Soft-Iron 완료")

# -------------------------------
# 3) 웨이브렛 (윈도우 방식으로 통일)
#    strategy = 'adaptive' or 'fixed'
# -------------------------------
strategy = 'adaptive'  # ← 필요시 'fixed'로
global_sigma = {c: None for c in mag_cols}

if strategy == 'fixed':
    # 학습 데이터에서 전역 sigma 추정 및 uthresh 산출/저장
    for c in mag_cols:
        s = df[c].values.astype(float)
        sigma = estimate_global_sigma(s, wavelet=WAVELET, level=LEVEL)
        # uthresh는 윈도우 길이에 의존 → WINDOW_SIZE 기준으로 고정
        uthresh = float(sigma * np.sqrt(2 * np.log(WINDOW_SIZE))) if sigma > 0 else 0.0
        global_sigma[c] = {'sigma': float(sigma), 'uthresh': uthresh}
    joblib.dump(global_sigma, 'wavelet_sigma.pkl')
    print("✅ Global sigma 저장:", global_sigma)

# 학습에서도 윈도우/홉으로 동일 처리 → 나중에 추론과 1:1 대응
denoised = {}
N = len(df)
num_win = max(1, (N - WINDOW_SIZE) // HOP_SIZE + 1)

for c in mag_cols:
    windows_out = []
    for i in range(num_win):
        start = i*HOP_SIZE
        end = start + WINDOW_SIZE
        if end > N:
            # 마지막은 패딩(경계 모드와 동일 효과) 또는 break
            segment = df[c].values[start:N]
            # 간단히 끝쪽을 반복 패딩
            pad_needed = WINDOW_SIZE - len(segment)
            if pad_needed > 0:
                segment = np.pad(segment, (0, pad_needed), mode='edge')
        else:
            segment = df[c].values[start:end]

        if strategy == 'adaptive':
            y = wavelet_denoise_window(segment)
        else:  # fixed
            uth = global_sigma[c]['uthresh'] if global_sigma[c] else None
            y = wavelet_denoise_window(segment, uthresh=uth)

        windows_out.append(y)

    den = overlap_add(windows_out, hop_size=HOP_SIZE, total_len=N)
    denoised[c] = den

# 디노이즈 반영
for c in mag_cols:
    df[c] = denoised[c]
print("✅ Wavelet(윈도우 방식) 완료")

# -------------------------------
# 4) StandardScaler (학습: fit, 추론: transform만)
# -------------------------------
scaler = StandardScaler()
df[all_sensor_cols] = scaler.fit_transform(df[all_sensor_cols])
print("✅ StandardScaler 완료")

# -------------------------------
# 5) 저장 (학습 아티팩트)
# -------------------------------
df.to_csv('preprocessed_data.csv', index=False)
joblib.dump(bias, 'bias.pkl')
joblib.dump(soft_iron_matrix, 'soft_iron_matrix.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 윈도우/웨이브렛 파라미터도 기록(재현성)
params = {
    'wavelet': WAVELET,
    'level': LEVEL,
    'thresh_mode': THRESH_MODE,
    'window_size': WINDOW_SIZE,
    'hop_size': HOP_SIZE,
    'border_mode': BORDER_MODE,
    'strategy': strategy
}
joblib.dump(params, 'preproc_params.pkl')
print("💾 저장 완료: preprocessed_data.csv, bias.pkl, soft_iron_matrix.pkl, scaler.pkl, preproc_params.pkl")
if strategy == 'fixed':
    print("💾 wavelet_sigma.pkl(축별 전역 sigma/uthresh)도 저장됨")
