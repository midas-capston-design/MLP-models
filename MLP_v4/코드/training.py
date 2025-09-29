import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# -------------------------------
# 0️⃣ 전처리된 CSV 불러오기
# -------------------------------
df = pd.read_csv('preprocessed_data.csv')
print("✅ 전처리 데이터 불러오기 완료!")
print(f"데이터 크기: {df.shape}")

# -------------------------------
# 1️⃣ 입력(X) / 라벨(y) 분리 (Mag + Ori 전체 6개 사용)
# -------------------------------
X = df[['Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z']].values  # 🔥 6개 입력
y = df['Position'].values

# -------------------------------
# 2️⃣ 라벨 인코딩
# -------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# -------------------------------
# 3️⃣ Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"📊 Train 샘플 수: {len(X_train)}, Test 샘플 수: {len(X_test)}")

# -------------------------------
# 4️⃣ MLP 학습
# -------------------------------
mlp = MLPClassifier(hidden_layer_sizes=(512, 256,128),   # 👉 기본 64-64 구조 유지
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    shuffle=True,
                    verbose=True,      # ✅ 학습 로그 출력
                    random_state=42)

print("\n🚀 MLP 모델 학습 시작...\n")
mlp.fit(X_train, y_train)
print("\n✅ MLP 학습 완료!\n")

# -------------------------------
# 5️⃣ 정확도 평가
# -------------------------------
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ 최종 Test Accuracy: {acc:.2%}")

# -------------------------------
# 6️⃣ 모델 & 라벨 인코더 저장
# -------------------------------
joblib.dump(mlp, 'mlp_model_6input.pkl')
joblib.dump(le, 'label_encoder_6input.pkl')

print("\n💾 mlp_model_6input.pkl, label_encoder_6input.pkl 저장 완료!")
