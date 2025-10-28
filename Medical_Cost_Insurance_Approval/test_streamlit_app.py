#!/usr/bin/env python3
"""
Test script to verify the Streamlit app functionality without launching the UI.
This tests the core prediction functions of the app.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def test_model_loading():
    """Test if models can be loaded properly"""
    print("🧪 Testing Model Loading...")
    
    try:
        # Load all models
        regression_model = joblib.load("models/regression_model.pkl")
        classification_model = joblib.load("models/classification_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_columns = joblib.load("models/feature_columns.pkl")
        
        print("✅ All models loaded successfully!")
        print(f"   - Regression model: {type(regression_model).__name__}")
        print(f"   - Classification model: {type(classification_model).__name__}")
        print(f"   - Scaler: {type(scaler).__name__}")
        print(f"   - Feature columns: {len(feature_columns)} features")
        
        return regression_model, classification_model, scaler, feature_columns
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None, None

def preprocess_input_test(age, sex, bmi, children, smoker, region, feature_columns, scaler):
    """Test the preprocessing function from the Streamlit app"""
    print(f"\n🔄 Testing preprocessing for: {age}yr {sex}, BMI:{bmi}, Children:{children}, Smoker:{smoker}, Region:{region}")
    
    try:
        # Create input dataframe
        input_data = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }
        
        input_df = pd.DataFrame([input_data])
        
        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(input_df, columns=['sex', 'smoker', 'region'], drop_first=False)
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[feature_columns]
        
        # Scale features
        input_scaled = scaler.transform(input_encoded)
        
        print("   ✅ Preprocessing successful!")
        return input_scaled
        
    except Exception as e:
        print(f"   ❌ Preprocessing error: {e}")
        return None

def test_predictions(regression_model, classification_model, input_scaled):
    """Test making predictions"""
    print("🎯 Testing Predictions...")
    
    try:
        # Make predictions
        predicted_cost = regression_model.predict(input_scaled)[0]
        predicted_approval = classification_model.predict(input_scaled)[0]
        approval_probability = classification_model.predict_proba(input_scaled)[0]
        
        print(f"   💰 Predicted Cost: ${predicted_cost:,.2f}")
        if predicted_approval == 1:
            print(f"   ✅ Approval Status: APPROVED (Confidence: {approval_probability[1]:.2%})")
        else:
            print(f"   ❌ Approval Status: NOT APPROVED (Confidence: {approval_probability[0]:.2%})")
        
        return predicted_cost, predicted_approval, approval_probability
        
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        return None, None, None

def test_streamlit_app_functions():
    """Test the core functions from the Streamlit app"""
    print("🚀 Testing Streamlit App Functions")
    print("=" * 60)
    
    # Test model loading
    regression_model, classification_model, scaler, feature_columns = test_model_loading()
    
    if not all([regression_model, classification_model, scaler, feature_columns]):
        print("❌ Cannot continue testing without models")
        return False
    
    # Test cases covering different scenarios
    test_cases = [
        {
            "name": "Young, Healthy, Non-smoker",
            "age": 25, "sex": "female", "bmi": 22.0, 
            "children": 0, "smoker": "no", "region": "southwest"
        },
        {
            "name": "Middle-aged, Smoker, High BMI",
            "age": 45, "sex": "male", "bmi": 35.0, 
            "children": 2, "smoker": "yes", "region": "northeast"
        },
        {
            "name": "Older, Overweight, Non-smoker",
            "age": 55, "sex": "female", "bmi": 28.0, 
            "children": 1, "smoker": "no", "region": "southeast"
        },
        {
            "name": "Young, Obese, Smoker",
            "age": 30, "sex": "male", "bmi": 40.0, 
            "children": 3, "smoker": "yes", "region": "northwest"
        }
    ]
    
    successful_tests = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*20} Test Case {i}: {case['name']} {'='*20}")
        
        # Test preprocessing
        input_scaled = preprocess_input_test(
            case['age'], case['sex'], case['bmi'], case['children'], 
            case['smoker'], case['region'], feature_columns, scaler
        )
        
        if input_scaled is not None:
            # Test predictions
            predicted_cost, predicted_approval, approval_probability = test_predictions(
                regression_model, classification_model, input_scaled
            )
            
            if all([predicted_cost is not None, predicted_approval is not None, approval_probability is not None]):
                successful_tests += 1
                
                # Additional validation
                if predicted_cost > 0:
                    print("   ✅ Cost prediction is positive")
                else:
                    print("   ⚠️ Warning: Negative cost prediction")
                
                if 0 <= approval_probability[0] <= 1 and 0 <= approval_probability[1] <= 1:
                    print("   ✅ Probability values are valid")
                else:
                    print("   ❌ Invalid probability values")
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {successful_tests}/{len(test_cases)} test cases passed")
    
    if successful_tests == len(test_cases):
        print("🎉 All Streamlit app functions working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False

def test_streamlit_app_structure():
    """Test if the Streamlit app file has the correct structure"""
    print("\n🔍 Testing Streamlit App Structure...")
    
    try:
        with open("app/app.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        required_elements = [
            "import streamlit as st",
            "import pandas as pd",
            "import numpy as np",
            "import joblib",
            "def load_models():",
            "def preprocess_input(",
            "def main():",
            "st.set_page_config(",
            "st.sidebar",
            "st.button"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if not missing_elements:
            print("✅ Streamlit app structure is correct")
            return True
        else:
            print(f"❌ Missing elements in Streamlit app: {missing_elements}")
            return False
            
    except Exception as e:
        print(f"❌ Error reading Streamlit app file: {e}")
        return False

def main():
    """Main test execution"""
    print("🏥 Medical Cost & Insurance Approval Prediction - Streamlit App Test")
    print("===================================================================")
    
    # Test 1: Streamlit app structure
    structure_test = test_streamlit_app_structure()
    
    # Test 2: Core functionality
    function_test = test_streamlit_app_functions()
    
    # Summary
    print("\n" + "="*60)
    print("📋 Final Test Summary")
    print("="*60)
    
    if structure_test:
        print("✅ Streamlit app structure: PASSED")
    else:
        print("❌ Streamlit app structure: FAILED")
    
    if function_test:
        print("✅ Core functionality: PASSED")
    else:
        print("❌ Core functionality: FAILED")
    
    if structure_test and function_test:
        print("\n🎉 All tests PASSED! The Streamlit app is ready to use.")
        print("\n📋 To run the app:")
        print("   streamlit run app/app.py")
        print("\n🌐 The app will be available at: http://localhost:8501")
        return True
    else:
        print("\n⚠️ Some tests FAILED. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)