#!/usr/bin/env python3
"""
Model Training Script - Medical Cost & Insurance Approval Prediction
This script performs the same operations as the Jupyter notebook but in a standalone Python script.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🏥 {title}")
    print(f"{'='*60}")

def load_and_inspect_data():
    """Load and inspect the dataset"""
    print_section("Data Loading and Inspection")
    
    # Load the dataset
    df = pd.read_csv('data/insurance.csv')
    
    print(f"🔍 Dataset Shape: {df.shape}")
    print(f"\n📋 First 5 rows:")
    print(df.head())
    
    print(f"\n📊 Dataset Info:")
    df.info()
    
    print(f"\n📈 Statistical Summary:")
    print(df.describe())
    
    # Check for missing values
    print(f"\n❓ Missing Values:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("✅ No missing values found!")
    else:
        print(f"⚠️ Total missing values: {missing_values.sum()}")
    
    # Check unique values for categorical columns
    print(f"\n🏷️ Unique values in categorical columns:")
    categorical_cols = ['sex', 'smoker', 'region']
    
    for col in categorical_cols:
        print(f"\n{col}: {df[col].unique()}")
        print(f"Value counts:\n{df[col].value_counts()}")
    
    return df

def perform_eda(df):
    """Perform Exploratory Data Analysis"""
    print_section("Exploratory Data Analysis (EDA)")
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Correlation analysis
    print("📊 Computing correlations...")
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('🔥 Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Key Correlations with Charges:")
    charges_corr = correlation_matrix['charges'].sort_values(ascending=False)
    print(charges_corr)
    
    # Distribution plots
    print("\n📈 Creating distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📈 Distribution of Numerical Features', fontsize=16, fontweight='bold')
    
    # Age distribution
    axes[0, 0].hist(df['age'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    
    # BMI distribution
    axes[0, 1].hist(df['bmi'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('BMI Distribution')
    axes[0, 1].set_xlabel('BMI')
    axes[0, 1].set_ylabel('Frequency')
    
    # Children distribution
    axes[1, 0].hist(df['children'], bins=6, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Children Distribution')
    axes[1, 0].set_xlabel('Number of Children')
    axes[1, 0].set_ylabel('Frequency')
    
    # Charges distribution
    axes[1, 1].hist(df['charges'], bins=30, color='pink', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Charges Distribution')
    axes[1, 1].set_xlabel('Charges')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('plots/numerical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ EDA completed and plots saved to 'plots/' directory")

def preprocess_data(df):
    """Preprocess the data for model training"""
    print_section("Data Preprocessing")
    
    # Create approval_status column for classification
    df['approval_status'] = (df['charges'] < 15000).astype(int)
    
    print("💼 Approval Status Distribution:")
    print(df['approval_status'].value_counts())
    print(f"Approval Rate: {df['approval_status'].mean():.2%}")
    
    # One-hot encoding for categorical variables
    print("\n🔄 Performing One-hot Encoding...")
    
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df_processed, columns=['sex', 'smoker', 'region'], drop_first=False)
    
    print(f"✅ Original shape: {df.shape}")
    print(f"✅ After encoding shape: {df_encoded.shape}")
    print(f"\n🏷️ New columns added:")
    new_columns = set(df_encoded.columns) - set(df.columns)
    print(list(new_columns))
    
    # Prepare features and targets
    print("\n🎯 Preparing Features and Targets...")
    
    # Features (exclude charges and approval_status)
    feature_columns = [col for col in df_encoded.columns if col not in ['charges', 'approval_status']]
    X = df_encoded[feature_columns]
    
    # Targets
    y_regression = df_encoded['charges']  # For regression
    y_classification = df_encoded['approval_status']  # For classification
    
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Regression target shape: {y_regression.shape}")
    print(f"✅ Classification target shape: {y_classification.shape}")
    
    print(f"\n📋 Feature columns:")
    for i, col in enumerate(X.columns, 1):
        print(f"{i:2d}. {col}")
    
    return X, y_regression, y_classification, feature_columns

def split_and_scale_data(X, y_regression, y_classification):
    """Split and scale the data"""
    print_section("Data Splitting and Scaling")
    
    # Split data for regression
    print("🔀 Splitting Data for Regression...")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    print(f"✅ Training set: {X_train_reg.shape[0]} samples")
    print(f"✅ Test set: {X_test_reg.shape[0]} samples")
    
    # Split data for classification
    print("\n🔀 Splitting Data for Classification...")
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
    )
    
    print(f"✅ Training set: {X_train_clf.shape[0]} samples")
    print(f"✅ Test set: {X_test_clf.shape[0]} samples")
    print(f"✅ Training approval rate: {y_train_clf.mean():.2%}")
    print(f"✅ Test approval rate: {y_test_clf.mean():.2%}")
    
    # Feature scaling
    print("\n⚖️ Applying Feature Scaling...")
    
    scaler = StandardScaler()
    
    # Fit scaler on training data and transform both training and test sets
    X_train_reg_scaled = scaler.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler.transform(X_test_reg)
    
    X_train_clf_scaled = scaler.transform(X_train_clf)
    X_test_clf_scaled = scaler.transform(X_test_clf)
    
    print("✅ Feature scaling completed!")
    print(f"✅ Scaled training features shape: {X_train_reg_scaled.shape}")
    print(f"✅ Scaled test features shape: {X_test_reg_scaled.shape}")
    
    return (X_train_reg_scaled, X_test_reg_scaled, y_train_reg, y_test_reg,
            X_train_clf_scaled, X_test_clf_scaled, y_train_clf, y_test_clf, scaler)

def train_regression_model(X_train_reg_scaled, y_train_reg, X_test_reg_scaled, y_test_reg):
    """Train and evaluate the regression model"""
    print_section("Regression Model Training (Medical Cost Prediction)")
    
    # Train Linear Regression model
    print("🏋️ Training Linear Regression Model...")
    
    regression_model = LinearRegression()
    regression_model.fit(X_train_reg_scaled, y_train_reg)
    
    # Make predictions
    y_train_pred_reg = regression_model.predict(X_train_reg_scaled)
    y_test_pred_reg = regression_model.predict(X_test_reg_scaled)
    
    print("✅ Regression model training completed!")
    
    # Evaluate regression model
    print("\n📊 Regression Model Evaluation:")
    print("=" * 50)
    
    # Training metrics
    train_r2 = r2_score(y_train_reg, y_train_pred_reg)
    train_mae = mean_absolute_error(y_train_reg, y_train_pred_reg)
    train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_train_pred_reg))
    
    # Test metrics
    test_r2 = r2_score(y_test_reg, y_test_pred_reg)
    test_mae = mean_absolute_error(y_test_reg, y_test_pred_reg)
    test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred_reg))
    
    print(f"📈 Training Metrics:")
    print(f"   R² Score: {train_r2:.4f}")
    print(f"   MAE: ${train_mae:,.2f}")
    print(f"   RMSE: ${train_rmse:,.2f}")
    
    print(f"\n🎯 Test Metrics:")
    print(f"   R² Score: {test_r2:.4f}")
    print(f"   MAE: ${test_mae:,.2f}")
    print(f"   RMSE: ${test_rmse:,.2f}")
    
    print(f"\n📊 Model Performance Summary:")
    print(f"   The model explains {test_r2:.1%} of the variance in medical costs")
    print(f"   Average prediction error: ±${test_mae:,.0f}")
    
    return regression_model, test_r2, test_mae

def train_classification_model(X_train_clf_scaled, y_train_clf, X_test_clf_scaled, y_test_clf, feature_columns):
    """Train and evaluate the classification model"""
    print_section("Classification Model Training (Insurance Approval Prediction)")
    
    # Train Random Forest Classifier
    print("🌲 Training Random Forest Classifier...")
    
    classification_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    classification_model.fit(X_train_clf_scaled, y_train_clf)
    
    # Make predictions
    y_train_pred_clf = classification_model.predict(X_train_clf_scaled)
    y_test_pred_clf = classification_model.predict(X_test_clf_scaled)
    
    # Get prediction probabilities
    y_test_pred_proba = classification_model.predict_proba(X_test_clf_scaled)
    
    print("✅ Classification model training completed!")
    
    # Evaluate classification model
    print("\n📊 Classification Model Evaluation:")
    print("=" * 50)
    
    # Training accuracy
    train_accuracy = accuracy_score(y_train_clf, y_train_pred_clf)
    test_accuracy = accuracy_score(y_test_clf, y_test_pred_clf)
    
    print(f"📈 Training Accuracy: {train_accuracy:.4f} ({train_accuracy:.1%})")
    print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.1%})")
    
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test_clf, y_test_pred_clf, 
                              target_names=['Not Approved', 'Approved']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_clf, y_test_pred_clf)
    print(f"\n🎭 Confusion Matrix:")
    print(f"   True Negatives: {cm[0,0]}")
    print(f"   False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]}")
    print(f"   True Positives: {cm[1,1]}")
    
    # Feature importance analysis
    print(f"\n🌟 Top 5 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': classification_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
        print(f"{i}. {row['feature']}: {row['importance']:.4f}")
    
    return classification_model, test_accuracy

def save_models(regression_model, classification_model, scaler, feature_columns):
    """Save trained models and preprocessing objects"""
    print_section("Model Saving")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save regression model
    joblib.dump(regression_model, 'models/regression_model.pkl')
    print("✅ Regression model saved to 'models/regression_model.pkl'")
    
    # Save classification model
    joblib.dump(classification_model, 'models/classification_model.pkl')
    print("✅ Classification model saved to 'models/classification_model.pkl'")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Scaler saved to 'models/scaler.pkl'")
    
    # Save feature columns for reference
    joblib.dump(list(feature_columns), 'models/feature_columns.pkl')
    print("✅ Feature columns saved to 'models/feature_columns.pkl'")
    
    print("\n🎉 All models and preprocessing objects saved successfully!")

def test_saved_models(feature_columns):
    """Test the saved models with sample predictions"""
    print_section("Model Testing")
    
    print("🧪 Testing Saved Models with Sample Predictions...")
    
    # Load saved models
    loaded_reg_model = joblib.load('models/regression_model.pkl')
    loaded_clf_model = joblib.load('models/classification_model.pkl')
    loaded_scaler = joblib.load('models/scaler.pkl')
    loaded_features = joblib.load('models/feature_columns.pkl')
    
    print("✅ Models loaded successfully!")
    
    # Create sample test cases
    test_cases = [
        {"age": 25, "bmi": 22.0, "children": 0, "sex": "female", "smoker": "no", "region": "southwest"},
        {"age": 45, "bmi": 30.0, "children": 2, "sex": "male", "smoker": "yes", "region": "northeast"},
        {"age": 35, "bmi": 25.0, "children": 1, "sex": "female", "smoker": "no", "region": "northwest"}
    ]
    
    print("\n🎯 Sample Predictions:")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        # Create DataFrame for the test case
        test_df = pd.DataFrame([case])
        
        # One-hot encode
        test_encoded = pd.get_dummies(test_df, columns=['sex', 'smoker', 'region'], drop_first=False)
        
        # Ensure all columns are present
        for col in loaded_features:
            if col not in test_encoded.columns:
                test_encoded[col] = 0
        
        # Reorder columns to match training
        test_encoded = test_encoded[loaded_features]
        
        # Scale features
        test_scaled = loaded_scaler.transform(test_encoded)
        
        # Make predictions
        predicted_cost = loaded_reg_model.predict(test_scaled)[0]
        predicted_approval = loaded_clf_model.predict(test_scaled)[0]
        approval_proba = loaded_clf_model.predict_proba(test_scaled)[0]
        
        print(f"\n👤 Test Case {i}:")
        print(f"   Profile: {case['age']}yr {case['sex']}, BMI:{case['bmi']}, Children:{case['children']}, Smoker:{case['smoker']}, Region:{case['region']}")
        print(f"   💰 Predicted Cost: ${predicted_cost:,.2f}")
        if predicted_approval == 1:
            print(f"   ✅ Approval Status: APPROVED (Probability: {approval_proba[1]:.2%})")
        else:
            print(f"   ❌ Approval Status: NOT APPROVED (Probability: {approval_proba[0]:.2%})")
    
    print("\n🎉 Model testing completed successfully!")

def main():
    """Main execution function"""
    print("🏥 Medical Cost & Insurance Approval Prediction - Model Training")
    print("================================================================")
    
    try:
        # Step 1: Load and inspect data
        df = load_and_inspect_data()
        
        # Step 2: Perform EDA
        perform_eda(df)
        
        # Step 3: Preprocess data
        X, y_regression, y_classification, feature_columns = preprocess_data(df)
        
        # Step 4: Split and scale data
        (X_train_reg_scaled, X_test_reg_scaled, y_train_reg, y_test_reg,
         X_train_clf_scaled, X_test_clf_scaled, y_train_clf, y_test_clf, scaler) = split_and_scale_data(
            X, y_regression, y_classification)
        
        # Step 5: Train regression model
        regression_model, test_r2, test_mae = train_regression_model(
            X_train_reg_scaled, y_train_reg, X_test_reg_scaled, y_test_reg)
        
        # Step 6: Train classification model
        classification_model, test_accuracy = train_classification_model(
            X_train_clf_scaled, y_train_clf, X_test_clf_scaled, y_test_clf, feature_columns)
        
        # Step 7: Save models
        save_models(regression_model, classification_model, scaler, feature_columns)
        
        # Step 8: Test saved models
        test_saved_models(feature_columns)
        
        # Final summary
        print_section("Training Complete!")
        print(f"🎯 Final Model Performance Summary:")
        print(f"   📊 Regression Model R² Score: {test_r2:.4f}")
        print(f"   📊 Regression Model MAE: ${test_mae:,.2f}")
        print(f"   📊 Classification Model Accuracy: {test_accuracy:.4f}")
        print(f"\n✅ All models trained and saved successfully!")
        print(f"✅ Ready to run Streamlit app: streamlit run app/app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
    
    print("\n🏥 Model training completed successfully! 🚀")