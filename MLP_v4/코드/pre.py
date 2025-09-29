import pandas as pd
import numpy as np
import pywt
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------
# 0️⃣ CSV 파일 열기
# -------------------------------
df = pd.read_csv('merged.csv')   # ✅ 앞/뒤 합친 원본 데이터
print("✅ 파일 불러오기 완료!")
print(f"데이터 크기: {df.shape}")
print(df.head())

# -------------------------------
# 1️⃣ 센서 컬럼 지정
# -------------------------------
mag_cols = ['Mag_X', 'Mag_Y', 'Mag_Z']     # Magnetometer
ori_cols = ['Ori_X', 'Ori_Y', 'Ori_Z']     # Orientation (선택)
all_sensor_cols = mag_cols + ori_cols

# -------------------------------
# 2️⃣ Hard-Iron 보정 (Magnetometer)
#    각 축의 오프셋을 제거
# -------------------------------
bias = {}
for col in mag_cols:
    max_val = df[col].max()
    min_val = df[col].min()
    bias[col] = (max_val + min_val) / 2.0
    df[col] = df[col] - bias[col]
print("✅ Hard-Iron 보정 완료")

# -------------------------------
# 3️⃣ Soft-Iron 보정 (Ellipsoid → Sphere)
#    공분산 기반 선형 스케일/회전 보정
# -------------------------------
cov = np.cov(df[mag_cols].values.T)
eig_vals, eig_vecs = np.linalg.eigh(cov)
# 수치 안정성용 근사(아주 작은 고유값 보호)
eps = 1e-12
scale = np.diag(1.0 / np.sqrt(np.clip(eig_vals, eps, None)))
soft_iron_matrix = eig_vecs @ scale @ eig_vecs.T

# 적용
df[mag_cols] = (soft_iron_matrix @ df[mag_cols].values.T).T
print("✅ Soft-Iron 보정 완료")

# -------------------------------
# 4️⃣ 웨이브렛 Denoising (Magnetometer)
#    db4, level=3, soft-threshold(Donoho universal)
# -------------------------------
def wavelet_denoise(signal, wavelet='db4', level=3, mode='soft'):
    # DWT
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    # 노이즈 표준편차 추정 (최상위 detail 계수의 median absolute deviation)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) > 0 else 0.0
    if sigma == 0.0:
        return signal  # 노이즈 추정 불가 시 원신호 반환
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    # Approximation(A) 제외, Detail(D) 계수만 임계값 적용
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]
    ]
    recon = pywt.waverec(denoised_coeffs, wavelet=wavelet)
    # 길이 보정 (경계 처리로 길이가 1~2 샘플 달라질 수 있음)
    return recon[:len(signal)]

for col in mag_cols:
    df[col] = wavelet_denoise(df[col].values, wavelet='db4', level=3, mode='soft')
print("✅ Wavelet Denoising 완료")

# -------------------------------
# 5️⃣ StandardScaler (표준화: 평균0, 분산1)
#    * Zero-Centering은 표준화에 포함되므로 생략
#    * 훈련셋으로 fit → 추론 시 transform만 사용
# -------------------------------
scaler = StandardScaler()
df[all_sensor_cols] = scaler.fit_transform(df[all_sensor_cols])
print("✅ StandardScaler 적용 완료")

# -------------------------------
# 6️⃣ 전처리 데이터 저장
# -------------------------------
df.to_csv('preprocessed_data.csv', index=False)
print("💾 'preprocessed_data.csv' 저장 완료!")

# -------------------------------
# 7️⃣ 전처리 파라미터 저장
# -------------------------------
joblib.dump(bias, 'bias.pkl')
joblib.dump(soft_iron_matrix, 'soft_iron_matrix.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n💾 bias.pkl, soft_iron_matrix.pkl, scaler.pkl 저장 완료!")
