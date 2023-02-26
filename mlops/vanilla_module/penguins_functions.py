import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def ingest_data():
    penguins = load_penguins()
    df = penguins.dropna()
    return df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]


def preprocess_data(df):
    # Feature engineering
    df['bill_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']
    df['body_mass_log'] = np.log(df['body_mass_g'])

    # X, y 구분
    X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'bill_ratio', 'body_mass_log']]
    y = df['sex']

    # 훈련/시험 데이터셋
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Engineering
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_data(X_train, X_test, y_train, y_test):
    # 데이터 적합 / 기계학습
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    # 평가
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # 평가결과 파일 저장
    with open('mlops/vanilla_module/results.txt', 'w') as f:
        f.write(f'함수 코딩 Accuracy: {accuracy}')

    return rfc


if __name__ == '__main__':
  
    df = ingest_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    train_data(X_train, X_test, y_train, y_test)


