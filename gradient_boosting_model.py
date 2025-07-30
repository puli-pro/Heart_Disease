"""
Heart Disease Prediction - Model 3: Gradient Boosting with Neural Network Ensemble
File: gradient_boosting_model.py

This model combines XGBoost, LightGBM, and Neural Networks in an ensemble approach
with advanced preprocessing, cross-validation, and deployment capabilities.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, roc_auc_score, roc_curve, log_loss)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class HeartDiseaseEnsembleModel:
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.calibrated_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.y = None
        self.data = None

    def load_and_comprehensive_analysis(self, file_path):
        """Load data with comprehensive statistical analysis"""
        print("=" * 70)
        print("HEART DISEASE PREDICTION - ADVANCED ENSEMBLE MODEL")
        print("=" * 70)

        self.data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully! Shape: {self.data.shape}")

        print(f"\n📊 COMPREHENSIVE DATA ANALYSIS")
        print("=" * 50)
        print(f"Dataset dimensions: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
        missing_analysis = self.data.isnull().sum()
        if missing_analysis.sum() == 0:
            print("✅ No missing values detected")
        else:
            print("⚠️ Missing values found:")
            for col, count in missing_analysis[missing_analysis > 0].items():
                print(f"  {col}: {count} ({count/len(self.data)*100:.2f}%)")

        target_analysis = self.data['HeartDisease'].value_counts()
        print(f"\n🎯 Target Variable Analysis:")
        print(f"  Class 0 (No Disease): {target_analysis[0]} ({target_analysis[0]/len(self.data)*100:.1f}%)")
        print(f"  Class 1 (Disease): {target_analysis[1]} ({target_analysis[1]/len(self.data)*100:.1f}%)")
        return self.data

    def advanced_feature_engineering(self):
        """State-of-the-art feature engineering"""
        print(f"\n🔧 ADVANCED FEATURE ENGINEERING")
        print("=" * 50)
        df = self.data.copy()
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        if (df['Cholesterol'] == 0).any():
            df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
            df['Cholesterol'].fillna(df.groupby(['Sex', 'Age'])['Cholesterol'].transform('median'), inplace=True)
            df['Cholesterol'].fillna(df['Cholesterol'].median(), inplace=True)
            print(f"   ✓ Imputed missing cholesterol values")

        df['Framingham_Risk_Factors'] = ((df['Age'] >= 45).astype(int) + (df['Sex'] == 1).astype(int) + 
                                       (df['Cholesterol'] >= 240).astype(int) + 
                                       (df['RestingBP'] >= 140).astype(int) + df['FastingBS'])
        df['Predicted_Max_HR'] = 220 - df['Age']
        df['HR_Achievement_Ratio'] = df['MaxHR'] / df['Predicted_Max_HR']
        df['Age_Cholesterol_Product'] = df['Age'] * df['Cholesterol'] / 1000
        df['Metabolic_Risk_Score'] = (df['Cholesterol'] / 200) + (df['RestingBP'] / 120) + df['FastingBS']
        
        self.X = df.drop('HeartDisease', axis=1)
        self.y = df['HeartDisease']
        self.feature_names = self.X.columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("\nData splitting and scaling complete.")

    def build_and_evaluate_models(self):
        """Build, train, and evaluate individual models."""
        print(f"\n🚀 MODEL TRAINING & EVALUATION")
        print("=" * 50)
        self.models = {
            "Logistic Regression": LogisticRegressionCV(random_state=42),
            "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": lgb.LGBMClassifier(random_state=42),
            "Neural Network": MLPClassifier(random_state=42, max_iter=500)
        }
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            print(f"\n--- {name} ---")
            print(f"  Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
            print(f"  F1-Score: {f1_score(self.y_test, y_pred):.4f}")

    def create_ensemble_and_evaluate(self):
        """Create and evaluate an advanced ensemble model."""
        print(f"\n👑 ENSEMBLE MODELING")
        print("=" * 50)
        estimators = list(self.models.items())
        meta_learner = LogisticRegressionCV(cv=5, random_state=42)
        self.ensemble_model = StackingClassifier(estimators=estimators, final_estimator=meta_learner, cv=5)
        self.ensemble_model.fit(self.X_train, self.y_train)
        self.calibrated_model = CalibratedClassifierCV(self.ensemble_model, method='isotonic', cv=5)
        self.calibrated_model.fit(self.X_train, self.y_train)
        y_pred = self.calibrated_model.predict(self.X_test)
        y_proba = self.calibrated_model.predict_proba(self.X_test)[:, 1]
        print(f"   ✅ Final Ensemble Performance:")
        print(f"      Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"      F1-Score: {f1_score(self.y_test, y_pred):.4f}")
        print(f"      ROC AUC: {roc_auc_score(self.y_test, y_proba):.4f}")

    def display_results(self):
        """Visualize model performance and insights."""
        print(f"\n🎨 VISUALIZING RESULTS")
        print("=" * 50)
        y_proba = self.calibrated_model.predict_proba(self.X_test)[:, 1]
        y_pred = self.calibrated_model.predict(self.X_test)

        plt.figure(figsize=(10, 7))
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        plt.plot(fpr, tpr, label=f'Ensemble ROC Curve (AUC = {roc_auc_score(self.y_test, y_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig("roc_curve.png")
        plt.close()
        print("   ✓ ROC curve saved as roc_curve.png")

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        plt.close()
        print("   ✓ Confusion matrix saved as confusion_matrix.png")

    def save_model(self, file_path="heart_disease_model.joblib"):
        """Save the entire model pipeline."""
        print(f"\n💾 SAVING MODEL PIPELINE")
        print("=" * 50)
        pipeline = {
            'model': self.calibrated_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(pipeline, file_path)
        print(f"   ✅ Model saved successfully to {file_path}")
