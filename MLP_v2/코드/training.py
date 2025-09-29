import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

# -------------------------------
# 0) CSV 로드
# -------------------------------
CSV_PATH = 'preprocessed_data.csv'

# Position이 문자열/숫자 섞여 있을 수 있어 string으로 강제
df = pd.read_csv(CSV_PATH, dtype={'Position': 'string'}, low_memory=False)
print("✅ 전처리 데이터 불러오기 완료!")
print(f"데이터 크기: {df.shape}")

# -------------------------------
# 1) 특징/라벨 준비: 숫자 강제 + 자기장 크기 추가
# -------------------------------
feature_cols_base = ['Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z']
label_col = 'Position'

# 숫자 강제 변환 (문자 섞여 있으면 NaN으로)
for c in feature_cols_base:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# 자기장 크기 |B| 추가
df['Mag_abs'] = np.sqrt(df['Mag_X']**2 + df['Mag_Y']**2 + df['Mag_Z']**2)

feature_cols = feature_cols_base + ['Mag_abs']  # 총 7개

# 유효 행만 필터링 (특징/라벨에 NaN 있으면 제거)
before = len(df)
df = df.dropna(subset=feature_cols + [label_col])
after = len(df)
if before != after:
    print(f"⚠️ NaN 제거로 {before - after}행 drop (남은 {after}행)")

X = df[feature_cols].values
y = df[label_col].astype(str).values

# -------------------------------
# 2) 라벨 인코딩
# -------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"🧾 클래스 수: {num_classes}")
print(f"예시 클래스들: {list(le.classes_)[:10]}{' ...' if num_classes>10 else ''}")

# -------------------------------
# 3) Train/Test Split
# -------------------------------
stratify_opt = y_encoded if num_classes > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=stratify_opt
)
print(f"📊 Train 샘플 수: {len(X_train)}, Test 샘플 수: {len(X_test)}")

# -------------------------------
# 4) 파이프라인(표준화 + MLP) 학습
# -------------------------------
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='tanh',
        solver='adam',
        max_iter=500,
        random_state=42,
        verbose=True
    ))
])

print("\n🚀 MLP 모델 학습 시작...\n")
pipeline.fit(X_train, y_train)
print("\n✅ MLP 학습 완료!\n")

# -------------------------------
# 5) 평가
# -------------------------------
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ 최종 Test Accuracy: {acc:.2%}")

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0, target_names=le.classes_))

# -------------------------------
# 6) 저장
# -------------------------------
joblib.dump(pipeline, 'mlp_pipeline_7input_magabs.pkl')
joblib.dump(le, 'label_encoder_6input.pkl')
pd.Series(feature_cols).to_csv('feature_columns_6input.csv', index=False, header=['feature'])

print("\n💾 저장 완료: mlp_pipeline_6input_magabs.pkl, label_encoder_6input.pkl, feature_columns_6input.csv")
