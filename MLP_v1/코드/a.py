import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------
# 0️⃣ CSV 파일 열기
# -------------------------------
df = pd.read_csv('merged_data.csv')   # ✅ 앞/뒤 합친 원본 데이터
print("✅ 파일 불러오기 완료!")
print(f"데이터 크기: {df.shape}")
print(df.head())

# -------------------------------
# 1️⃣ 센서 컬럼 지정
# -------------------------------
mag_cols = ['Mag_X', 'Mag_Y', 'Mag_Z']     # Magnetometer
ori_cols = ['Ori_X', 'Ori_Y', 'Ori_Z']     # Orientation
all_sensor_cols = mag_cols + ori_cols

# -------------------------------
# 2️⃣ Hard-Iron 보정 (Magnetometer)
# -------------------------------
bias = {}
for col in mag_cols:
    max_val = df[col].max()
    min_val = df[col].min()
    bias[col] = (max_val + min_val) / 2
    df[col] = df[col] - bias[col]
print("✅ Hard-Iron 보정 완료")

# -------------------------------
# 3️⃣ Soft-Iron 보정 (Ellipsoid fitting)
# -------------------------------
cov = np.cov(df[mag_cols].values.T)
eig_vals, eig_vecs = np.linalg.eigh(cov)
scale = np.diag(1.0 / np.sqrt(eig_vals))
soft_iron_matrix = eig_vecs @ scale @ eig_vecs.T

# 적용
df[mag_cols] = (soft_iron_matrix @ df[mag_cols].values.T).T
print("✅ Soft-Iron 보정 완료")

# -------------------------------
# 4️⃣ Zero-Centering (평균 0 맞추기)
# -------------------------------
zero_center_means = {}
for col in all_sensor_cols:
    zero_center_means[col] = df[col].mean()
    df[col] = df[col] - zero_center_means[col]
print("✅ Zero-Centering 완료")

# -------------------------------
# 5️⃣ StandardScaler (표준화)
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
joblib.dump(zero_center_means, 'zero_center_means.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n💾 bias.pkl, soft_iron_matrix.pkl, zero_center_means.pkl, scaler.pkl 저장 완료!")
