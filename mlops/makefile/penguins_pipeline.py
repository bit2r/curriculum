
import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score

penguins = load_penguins()

df = penguins.dropna()

# X, y 구분
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['sex']

# 훈련 시험데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 상수 feature 제거
selector = VarianceThreshold()
X_train = selector.fit_transform(X_train)
X_test = selector.transform(X_test)

# 파이프라인 구축
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('classifier', RandomForestClassifier())
])

# 기계학습과 예측
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
print('파이프라인 방식 정확도(Accuracy):', accuracy)

# 출력결과 내보내기
with open('results.txt', 'w') as f:
    f.write(f'\n파이프라인 방식 정확도(Accuracy): {accuracy}\n')
