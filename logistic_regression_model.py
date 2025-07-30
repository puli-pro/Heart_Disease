"""
Heart Disease Prediction - Model 1: Logistic Regression with SHAP Explainability
File: logistic_regression_model.py

This model uses Logistic Regression for heart disease prediction with comprehensive
data preprocessing, feature engineering, and SHAP explanations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve, log_loss,
                             classification_report)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import shap
import warnings

warnings.filterwarnings('ignore')

class HeartDiseaseLogisticModel:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.pipeline = None
        self.preprocessor = None
        self.feature_names = None
        self.sample_case = None # Store a sample case for explanation
        self.feature_names_before_preprocessing = None # Store feature names before preprocessing

        # Store medians for single case imputation if Cholesterol=0 is treated as missing
        self.cholesterol_median = None
        self.cholesterol_sex_age_medians = None # For more nuanced imputation

    def load_and_explore_data(self, file_path):
        """Load data and perform basic exploration."""
        print("=" * 60)
        print("HEART DISEASE PREDICTION - LOGISTIC REGRESSION MODEL")
        print("=" * 60)
        self.data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully!\nShape: {self.data.shape}")
        print("\nFirst 5 rows:\n", self.data.head())
        print("\nDataset Info:")
        self.data.info()
        print("\nMissing Values:\n", self.data.isnull().sum().sum())
        if self.data.isnull().sum().sum() == 0:
            print("No missing values found!")
        print("\nStatistical Summary:\n", self.data.describe())
        print("\nTarget Variable Distribution:\n", self.data['HeartDisease'].value_counts())
        print(f"Heart Disease Rate: {self.data['HeartDisease'].mean():.2%}")
        return self.data

    def preprocess_data(self):
        """Perform comprehensive data preprocessing including feature engineering."""
        print("\n" + "=" * 40)
        print("DATA PREPROCESSING")
        print("=" * 40)
        df = self.data.copy()
        target = 'HeartDisease'

        # Feature Engineering
        print("Feature Engineering...")

        # Impute 0 cholesterol values (assuming 0 is missing)
        if (df['Cholesterol'] == 0).any():
            # Calculate medians BEFORE replacement for training set consistency
            self.cholesterol_sex_age_medians = df.groupby(['Sex', 'Age'])['Cholesterol'].median().to_dict()
            self.cholesterol_median = df['Cholesterol'].median()

            df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
            
            # Use a helper function for imputation to avoid direct groupby transform in single case
            # This is a temporary measure for demonstration; ideally, a custom imputer in pipeline
            # df['Cholesterol'].fillna(df.groupby(['Sex', 'Age'])['Cholesterol'].transform('median'), inplace=True)
            # df['Cholesterol'].fillna(df['Cholesterol'].median(), inplace=True)
            
            # A more robust imputation for training, ensuring no NaNs for single case prediction
            # For simplicity, let's use the overall median if grouped median isn't available
            df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())

            print("  - Imputed 0 cholesterol values with median (overall or group based for training).")


        # Create interaction terms
        # Ensure 'Sex' is mapped to numeric if it's 'M'/'F' for multiplication
        df['Age_Sex_Interaction'] = df['Age'] * df['Sex'].map({'M': 1, 'F': 0}) 
        df['MaxHR_Oldpeak_Interaction'] = df['MaxHR'] * df['Oldpeak']

        # Polynomial features (example for Age)
        df['Age_Squared'] = df['Age']**2

        # Ratio features
        df['Cholesterol_MaxHR_Ratio'] = df['Cholesterol'] / (df['MaxHR'] + 1e-6) # Add small epsilon to avoid division by zero

        # Domain-specific features
        # Framingham Risk Score components (simplified)
        df['Framingham_Risk_Factors'] = (
            (df['Age'] >= 45).astype(int) +
            (df['Sex'] == 'M').astype(int) + # Keep Sex as 'M' for this check
            (df['Cholesterol'] >= 240).astype(int) +
            (df['RestingBP'] >= 140).astype(int) +
            df['FastingBS']
        )

        # Pulse Pressure (assuming RestingBP is systolic, need diastolic for true pulse pressure)
        df['PulsePressure'] = df['RestingBP'] 

        # Resting ECG abnormalities count
        df['ECG_Abnormalities'] = df['RestingECG'].apply(lambda x: 1 if x in ['ST', 'LVH'] else 0)

        # Chest Pain Severity (ordinal mapping)
        chest_pain_mapping = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
        # Apply map, then fillna for any unmapped values, then convert to int
        df['ChestPain_Severity'] = df['ChestPainType'].map(chest_pain_mapping).fillna(-1).astype(int)


        # ST Depression Category
        # Assign to a temporary series, fillna, then convert to int
        temp_st_depression = pd.cut(df['Oldpeak'], bins=[-np.inf, 0, 1, 2, np.inf], labels=[0, 1, 2, 3])
        df['ST_Depression_Category'] = temp_st_depression.cat.add_categories(-1).fillna(-1).astype(int)


        # Blood Pressure categories
        temp_bp_category = pd.cut(df['RestingBP'], bins=[0, 120, 140, 1000], labels=[0, 1, 2], right=False)
        df['BP_Category'] = temp_bp_category.cat.add_categories(-1).fillna(-1).astype(int)


        # Heart Rate categories
        temp_maxhr_category = pd.cut(df['MaxHR'], bins=[0, 100, 150, np.inf], labels=[0, 1, 2])
        df['MaxHR_Category'] = temp_maxhr_category.cat.add_categories(-1).fillna(-1).astype(int)


        print("  - Created various interaction, polynomial, ratio, and domain-specific features.")
        print(f"  - Total features after engineering: {df.shape[1] - 1}") # Subtract target column

        # Define features and target
        X = df.drop(target, axis=1)
        y = df[target]
        self.feature_names_before_preprocessing = X.columns.tolist() # Store original feature names for single case prediction

        # Define categorical and numerical features for preprocessing
        # Update categorical features with newly created ones that might be categorical
        categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        engineered_categorical_features = ['ECG_Abnormalities', 'ChestPain_Severity', 'ST_Depression_Category', 'BP_Category', 'MaxHR_Category']
        
        # Add engineered categorical features if they are not already in the original list
        for col in engineered_categorical_features:
            if col not in categorical_features and col in X.columns: # Check if column actually exists in X
                categorical_features.append(col)

        # Numerical features are all columns not in the final list of categorical features
        numerical_features = [col for col in X.columns if col not in categorical_features]
        
        # Double check for any remaining non-numeric types in numerical features (should ideally be handled)
        for col in numerical_features:
            if X[col].dtype == 'object':
                print(f"Warning: '{col}' is still an object type but specified as numerical. Attempting conversion.")
                X[col] = pd.to_numeric(X[col], errors='coerce') # Coerce errors to NaN
                X[col].fillna(X[col].median(), inplace=True) # Fill NaNs introduced by coercion

        # Create preprocessing pipelines for numerical and categorical features
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore') # Handle unseen categories during prediction

        # Create a column transformer to apply different transformations to different columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough' # Keep other columns (if any)
        )

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("\nData splitting complete.")

        # Fit and transform data using the preprocessor
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        print("Data preprocessing (scaling and encoding) complete.")

        # Get the feature names after preprocessing
        num_feature_names = numerical_features
        cat_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        
        # Handle cases where 'remainder' might add columns (e.g., if passthrough columns exist and are not num/cat)
        # In this specific case, 'remainder' should not add new columns as all are handled.
        # But for robustness, if you had 'passthrough' columns that weren't numerical_features or categorical_features
        # you'd need to get their names too. For this problem, it's fine.
        
        self.feature_names = np.concatenate([num_feature_names, cat_feature_names])

        # Store a sample case for SHAP explanation (using the first instance of the test set)
        if self.X_test.shape[0] > 0:
            self.sample_case = self.X_test[0, :].reshape(1, -1)
        else:
            self.sample_case = None
            print("Warning: Test set is empty, cannot create sample case for SHAP.")

        return self.X_train, self.y_train

    def train_model(self, X_train, y_train):
        """Train the Logistic Regression model with cross-validation."""
        print("\n" + "=" * 40)
        print("MODEL TRAINING")
        print("=" * 40)
        self.model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, C=0.1)

        self.model.fit(X_train, y_train)
        print("Logistic Regression model trained successfully.")

        cv_scores = cross_val_score(self.model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc')
        print(f"\nCross-validation ROC-AUC scores: {cv_scores}")
        print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        self.model = CalibratedClassifierCV(self.model, method='isotonic', cv=5)
        self.model.fit(X_train, y_train)
        print("Model calibrated successfully.")

        return self.model

    def evaluate_model(self):
        """Evaluate the trained model on the test set."""
        print("\n" + "=" * 40)
        print("MODEL EVALUATION")
        print("=" * 40)
        if self.model is None or self.X_test is None or self.y_test is None:
            print("Model not trained or test data not available.")
            return None

        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        logloss = log_loss(self.y_test, y_proba)
        cm = confusion_matrix(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'log_loss': logloss,
            'confusion_matrix': cm,
            'classification_report': report
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()

        plt.subplot(1, 3, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        plt.subplot(1, 3, 3)
        sns.histplot(y_proba[self.y_test == 0], color='blue', label='No Disease', kde=True, stat='density', common_norm=False)
        sns.histplot(y_proba[self.y_test == 1], color='red', label='Disease', kde=True, stat='density', common_norm=False)
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return metrics

    def explain_with_shap(self):
        """Generate SHAP explanations for model interpretability."""
        print("\n" + "=" * 40)
        print("SHAP EXPLANATIONS")
        print("=" * 40)
        if self.model is None or self.X_test is None or self.feature_names is None or self.sample_case is None:
            print("Model not trained, test data, feature names, or sample case not available.")
            return

        try:
            explainer = shap.Explainer(self.model.predict_proba, self.X_test, feature_names=self.feature_names)
            shap_values = explainer(self.X_test)

            print("\nSHAP Summary Plot:")
            shap.summary_plot(shap_values[:, :, 1], self.X_test, feature_names=self.feature_names, plot_type='bar', show=False)
            plt.title("SHAP Feature Importance")
            plt.show()

            shap.summary_plot(shap_values[:, :, 1], self.X_test, feature_names=self.feature_names, show=False)
            plt.title("SHAP Summary Plot (beeswarm)")
            plt.show()

            if self.sample_case is not None:
                print("\nSHAP Explanation for a Sample Case:")
                sample_shap_values = explainer(self.sample_case)
                shap.waterfall_plot(sample_shap_values[0, :, 1], show=False, max_display=15, feature_names=self.feature_names)
                plt.title("SHAP Waterfall Plot for a Sample Case")
                plt.show()

        except Exception as e:
            print(f"Error during SHAP explanation: {e}")
            print("Please ensure shap is installed (`pip install shap`) and the model/data are correctly formatted.")

    def _apply_feature_engineering_to_single_case(self, df_single):
        """Helper to apply feature engineering consistently for single case prediction."""
        # This function must exactly replicate the feature engineering in preprocess_data
        
        # Impute 0 cholesterol values (if treated as missing during training)
        if 'Cholesterol' in df_single.columns and (df_single['Cholesterol'] == 0).any():
            df_single['Cholesterol'] = df_single['Cholesterol'].replace(0, np.nan)
            # Use the overall median calculated from the training data
            if self.cholesterol_median is not None:
                df_single['Cholesterol'].fillna(self.cholesterol_median, inplace=True)
            else:
                # Fallback if no median was stored (shouldn't happen if model was trained)
                print("Warning: Cholesterol median not available for single case imputation.")
                df_single['Cholesterol'].fillna(df_single['Cholesterol'].mean(), inplace=True) # Use its own mean as last resort

        df_single['Age_Sex_Interaction'] = df_single['Age'] * df_single['Sex'].map({'M': 1, 'F': 0})
        df_single['MaxHR_Oldpeak_Interaction'] = df_single['MaxHR'] * df_single['Oldpeak']
        df_single['Age_Squared'] = df_single['Age']**2
        df_single['Cholesterol_MaxHR_Ratio'] = df_single['Cholesterol'] / (df_single['MaxHR'] + 1e-6)

        df_single['Framingham_Risk_Factors'] = (
            (df_single['Age'] >= 45).astype(int) +
            (df_single['Sex'] == 'M').astype(int) +
            (df_single['Cholesterol'] >= 240).astype(int) +
            (df_single['RestingBP'] >= 140).astype(int) +
            df_single['FastingBS']
        )
        df_single['PulsePressure'] = df_single['RestingBP']
        df_single['ECG_Abnormalities'] = df_single['RestingECG'].apply(lambda x: 1 if x in ['ST', 'LVH'] else 0)

        chest_pain_mapping = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
        df_single['ChestPain_Severity'] = df_single['ChestPainType'].map(chest_pain_mapping).fillna(-1).astype(int)

        temp_st_depression = pd.cut(df_single['Oldpeak'], bins=[-np.inf, 0, 1, 2, np.inf], labels=[0, 1, 2, 3])
        df_single['ST_Depression_Category'] = temp_st_depression.cat.add_categories(-1).fillna(-1).astype(int)

        temp_bp_category = pd.cut(df_single['RestingBP'], bins=[0, 120, 140, 1000], labels=[0, 1, 2], right=False)
        df_single['BP_Category'] = temp_bp_category.cat.add_categories(-1).fillna(-1).astype(int)

        temp_maxhr_category = pd.cut(df_single['MaxHR'], bins=[0, 100, 150, np.inf], labels=[0, 1, 2])
        df_single['MaxHR_Category'] = temp_maxhr_category.cat.add_categories(-1).fillna(-1).astype(int)
        
        return df_single

    def predict_single_case(self, case_data=None):
        """
        Predict heart disease for a single case with explanation.

        Args:
            case_data (dict or pd.DataFrame): A dictionary or DataFrame
                                                representing the single case.
                                                If None, uses the internal sample_case.
        """
        print("\n" + "=" * 40)
        print("SINGLE CASE PREDICTION")
        print("=" * 40)

        if self.model is None or self.preprocessor is None or self.feature_names is None or self.feature_names_before_preprocessing is None:
            print("Model, preprocessor, feature names, or original feature names not available.")
            return

        if case_data is None:
            if self.sample_case is not None:
                print("Using internal sample case for prediction.")
                input_data_processed = self.sample_case
                # To display the raw data for the sample case, we'd need to store the original test set row
                # Let's retrieve it from the original dataframe based on the first test set index
                original_index = self.y_test.index[0]
                raw_input_data_display = self.data.loc[[original_index]].drop('HeartDisease', axis=1) # Drop target for display
            else:
                print("No sample case available and no case_data provided.")
                return
        else:
            print("Using provided case data for prediction.")
            try:
                raw_input_data = pd.DataFrame([case_data])
                raw_input_data_display = raw_input_data.copy()

                # Apply feature engineering to the single case
                df_single_engineered = self._apply_feature_engineering_to_single_case(raw_input_data.copy())

                # Ensure columns are in the same order as during training
                # This is CRUCIAL for ColumnTransformer
                df_single_engineered = df_single_engineered[self.feature_names_before_preprocessing]

                # Apply the preprocessor
                input_data_processed = self.preprocessor.transform(df_single_engineered)

            except Exception as e:
                print(f"Error preprocessing single case data: {e}")
                print("Please ensure the input data dictionary matches the expected features and types.")
                return

        proba = self.model.predict_proba(input_data_processed)[:, 1]
        prediction = self.model.predict(input_data_processed)[0]

        print(f"\nInput Data:\n{raw_input_data_display}")
        print(f"\nPredicted Probability of Heart Disease (Class 1): {proba[0]:.4f}")
        print(f"Predicted Class: {prediction}")

        try:
            explainer = shap.Explainer(self.model.predict_proba, self.X_train, feature_names=self.feature_names)
            shap_values_single = explainer(input_data_processed)

            print("\nSHAP Explanation for this Prediction:")
            shap.waterfall_plot(shap_values_single[0, :, 1], show=False, max_display=15, feature_names=self.feature_names)
            plt.title("SHAP Waterfall Plot for Single Prediction")
            plt.show()
        except Exception as e:
            print(f"Error generating SHAP explanation for single case: {e}")

    def save_model(self, file_path="logistic_regression_heart_model.joblib"):
        """Save the trained model and preprocessor."""
        print("\n" + "=" * 40)
        print("SAVING MODEL")
        print("=" * 40)
        if self.model is None or self.preprocessor is None:
            print("Model or preprocessor not trained/fitted. Cannot save.")
            return

        try:
            pipeline_to_save = {
                'model': self.model,
                'preprocessor': self.preprocessor,
                'feature_names': self.feature_names,
                'feature_names_before_preprocessing': self.feature_names_before_preprocessing,
                'cholesterol_median': self.cholesterol_median, # Save for single case imputation
                'cholesterol_sex_age_medians': self.cholesterol_sex_age_medians # Save for single case imputation
            }
            joblib.dump(pipeline_to_save, file_path)
            print(f"Model pipeline saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, file_path="logistic_regression_heart_model.joblib"):
        """Load a saved model pipeline."""
        print("\n" + "=" * 40)
        print("LOADING MODEL")
        print("=" * 40)
        try:
            loaded_pipeline = joblib.load(file_path)
            self.model = loaded_pipeline['model']
            self.preprocessor = loaded_pipeline['preprocessor']
            self.feature_names = loaded_pipeline['feature_names']
            self.feature_names_before_preprocessing = loaded_pipeline['feature_names_before_preprocessing']
            self.cholesterol_median = loaded_pipeline.get('cholesterol_median')
            self.cholesterol_sex_age_medians = loaded_pipeline.get('cholesterol_sex_age_medians')
            print(f"Model pipeline loaded successfully from {file_path}")
            return True
        except FileNotFoundError:
            print(f"Error: Model file not found at {file_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Main execution function"""
    # Initialize model
    model = HeartDiseaseLogisticModel()
    
    # Load and explore data
    data = model.load_and_explore_data('heart.csv')
    
    # Preprocess data
    X_train_processed, y_train = model.preprocess_data()
    
    # Train model
    trained_model = model.train_model(X_train_processed, y_train)
    
    # Evaluate model
    metrics = model.evaluate_model()
    
    # SHAP explanations
    model.explain_with_shap()
    
    # Single case prediction (using the internal sample case from test set)
    model.predict_single_case(None)
    
    # Example of predicting a new, unseen case (you'd need to provide real values)
    # Ensure this data mimics the structure and types of your *original* dataset
    # For example, 'Sex': 'M', 'ChestPainType': 'ATA', etc.
    new_patient_data = {
        'Age': 55,
        'Sex': 'F',
        'ChestPainType': 'NAP',
        'RestingBP': 125,
        'Cholesterol': 230,
        'FastingBS': 0,
        'RestingECG': 'Normal',
        'MaxHR': 160,
        'ExerciseAngina': 'N',
        'Oldpeak': 0.5,
        'ST_Slope': 'Flat'
    }
    model.predict_single_case(new_patient_data)

    # Save the trained model pipeline
    model.save_model("logistic_regression_heart_model.joblib")

    # Demonstrate loading the model and making a prediction with the loaded model
    # loaded_model_instance = HeartDiseaseLogisticModel()
    # if loaded_model_instance.load_model("logistic_regression_heart_model.joblib"):
    #     print("\nPrediction with loaded model:")
    #     loaded_model_instance.predict_single_case(new_patient_data)
    
    print("\n" + "=" * 60)
    print("MODEL 1 COMPLETED SUCCESSFULLY!")
    print("Key Features:")
    print("- Logistic Regression with cross-validation and calibration")
    print("- Robust feature engineering with ColumnTransformer and OneHotEncoder")
    print("- SHAP explanations for interpretability")
    print("- Comprehensive evaluation and visualizations")
    print("- Single case prediction with detailed explanation")
    print("=" * 60)

if __name__ == "__main__":
    main()