
import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

penguins = load_penguins()

df = penguins.dropna()

# X, y 구분
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['sex']

# 훈련 시험데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RF 분류모형
rfc = RandomForestClassifier()

# 기계학습
rfc.fit(X_train, y_train)

# 예측값 생성
y_pred = rfc.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
print('정확도(Accuracy):', accuracy)


