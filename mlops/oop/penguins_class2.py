import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score


class PenguinSexClassifier:
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = None
        
    def load_data(self):
        # Load data
        penguins = load_penguins()

        # Drop rows with missing values
        df = penguins.dropna()

        # Define features and target variable
        X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
        y = df['sex']

        # Split data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
    
    def preprocess_data(self):
        # Remove constant features using VarianceThreshold
        selector = VarianceThreshold()
        self.X_train = selector.fit_transform(self.X_train)
        self.X_test = selector.transform(self.X_test)
        
        # Scale features using StandardScaler
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_data(self):
        # Define pipeline with RandomForestClassifier as estimator
        self.pipeline = Pipeline([
            ('classifier', RandomForestClassifier())
        ])

        # Train model
        self.pipeline.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        # Make predictions
        self.y_pred = self.pipeline.predict(self.X_test)
        
        # Evaluate accuracy of predictions
        self.accuracy = accuracy_score(self.y_test, self.y_pred)

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.train_data()
        self.evaluate()
        print('정확도(Accuracy):', self.accuracy)

if __name__ == '__main__':
    clf = PenguinSexClassifier()
    clf.run()

