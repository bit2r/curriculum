import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score

class PenguinSexClassifier:
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline = None
        
    def load_data(self):
      
        penguins = load_penguins()

        df = penguins.dropna()

        # X, y 구분
        self.X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
        self.y = df['sex']
    
    def preprocess_data(self):
        # 훈련 시험데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        # 상수 Feature 제거
        selector = VarianceThreshold()
        X_train = selector.fit_transform(X_train)
        X_test = selector.transform(X_test)

        # 파이프라인 구축
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)),            
            ('classifier', RandomForestClassifier())
        ])

        # 모형 적합(훈련)
        self.pipeline.fit(X_train, y_train)
        
        # 모형 예측
        self.y_pred = self.pipeline.predict(X_test)
        
        # 예측 정확도 평가
        self.accuracy = accuracy_score(y_test, self.y_pred)

    def run(self):
        self.load_data()
        self.preprocess_data()

if __name__ == '__main__':
    clf = PenguinSexClassifier()
    clf.run()
    print('OOP 정확도(Accuracy):', clf.accuracy)
    
