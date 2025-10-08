# train.py
# -*- coding: utf-8 -*-
"""
학습 코드 (온라인 전처리와 100% 호환)
- 입력: preprocessed_data.csv  (이미 Hard/Soft-Iron, Windowed Wavelet, StandardScaler 적용)
- 출력: mlp_model_6input.pkl, label_encoder_6input.pkl  (model/ 폴더에 저장)
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

# -------------------------------
# 경로 설정
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

PREPROC_CSV = os.path.join(BASE_DIR, "preprocessed_data.csv")
REQUIRED_PREPROC_ARTIFACTS = [
    os.path.join(BASE_DIR, "bias.pkl"),
    os.path.join(BASE_DIR, "soft_iron_matrix.pkl"),
    os.path.join(BASE_DIR, "scaler.pkl"),
    os.path.join(BASE_DIR, "preproc_params.pkl"),
]
# fixed 전략이면 wavelet_sigma.pkl도 있을 수 있음(있으면 그대로 MODEL_DIR로 복사 권장)
WAVELET_SIGMA_PATH = os.path.join(BASE_DIR, "wavelet_sigma.pkl")

# -------------------------------
# 사전 체크 (전처리 산출물 존재여부)
# -------------------------------
if not os.path.exists(PREPROC_CSV):
    print("❌ 'preprocessed_data.csv'가 없습니다. 먼저 전처리 스크립트를 실행하세요.")
    sys.exit(1)

missing = [p for p in REQUIRED_PREPROC_ARTIFACTS if not os.path.exists(p)]
if missing:
    print("⚠️ 전처리 아티팩트 일부가 없습니다(학습은 진행되지만, 서버 호환 위해 준비 권장):")
    for m in missing:
        print("  -", os.path.basename(m))
else:
    print("✅ 전처리 아티팩트 존재 확인 완료.")

# -------------------------------
# 0) 데이터 로드
# -------------------------------
df = pd.read_csv(PREPROC_CSV)
print("✅ 전처리 데이터 불러오기 완료:", df.shape)

# 필수 컬럼 정의 (서버 FEATURE_COLS와 동일)
FEATURE_COLS = ['Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z']
LABEL_COL = 'Position'

# 방어적 체크
for c in FEATURE_COLS + [LABEL_COL]:
    if c not in df.columns:
        raise KeyError(f"필수 컬럼 누락: {c}")

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[LABEL_COL].astype(str).values

# -------------------------------
# 1) 라벨 인코딩
# -------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"🔤 클래스 개수: {num_classes}")

# -------------------------------
# 2) Train/Test Split (stratify)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"📊 Train: {len(X_train)} / Test: {len(X_test)}")

# -------------------------------
# 3) MLP 학습
#    - 입력은 이미 표준화되어 있으므로 추가 스케일링 불필요
#    - 조기 종료 켜서 과적합 방지
# -------------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    max_iter=500,
    shuffle=True,
    verbose=True,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=20,
    validation_fraction=0.1,
)

print("\n🚀 MLP 모델 학습 시작...\n")
mlp.fit(X_train, y_train)
print("\n✅ MLP 학습 완료!\n")

# -------------------------------
# 4) 평가
# -------------------------------
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc:.2%}")

# Top-3 accuracy (서버 측 Top-3와 개념 정합)
try:
    proba = mlp.predict_proba(X_test)
    top3_acc = top_k_accuracy_score(y_test, proba, k=3, labels=np.arange(num_classes))
    print(f"✅ Top-3 Accuracy: {top3_acc:.2%}")
except Exception:
    print("ℹ️ predict_proba 미지원 또는 오류로 Top-3 계산 생략")

# 클래스별 리포트(요약)
print("\n📄 Classification Report(요약):")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# -------------------------------
# 5) 저장 (서버 호환 파일명/경로)
# -------------------------------
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model_6input.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_6input.pkl")

joblib.dump(mlp, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"\n💾 저장 완료: {MODEL_PATH}, {ENCODER_PATH}")

# 전처리 아티팩트도 model 폴더에 복사(서버가 model/에서 읽는 경우 대비)
for src in REQUIRED_PREPROC_ARTIFACTS:
    if os.path.exists(src):
        dst = os.path.join(MODEL_DIR, os.path.basename(src))
        if src != dst:
            joblib.dump(joblib.load(src), dst)
            print(f"📦 {os.path.basename(src)} → model/ 복사 완료")

# fixed 전략용 wavelet_sigma도 있으면 함께 복사
if os.path.exists(WAVELET_SIGMA_PATH):
    dst = os.path.join(MODEL_DIR, "wavelet_sigma.pkl")
    if WAVELET_SIGMA_PATH != dst:
        joblib.dump(joblib.load(WAVELET_SIGMA_PATH), dst)
    print("📦 wavelet_sigma.pkl → model/ 복사 완료")

print("\n🎉 All done. 서버 코드와 100% 호환되는 모델/아티팩트가 준비되었습니다.")
