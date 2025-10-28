#!/usr/bin/env python3
"""
Quick test script to verify the project setup works correctly.
This script tests data loading and basic functionality before running the full notebook.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_project_structure():
    """Test if all required directories and files exist"""
    print("🔍 Testing project structure...")
    
    required_files = [
        "data/insurance.csv",
        "app/app.py", 
        "notebooks/EDA_and_Model_Training.ipynb",
        "requirements.txt",
        "README.md"
    ]
    
    required_dirs = [
        "data",
        "app",
        "notebooks", 
        "models"
    ]
    
    # Check directories
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"❌ Missing directory: {directory}")
            return False
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            return False
    
    return True

def test_data_loading():
    """Test if the dataset can be loaded properly"""
    print("\n📊 Testing data loading...")
    
    try:
        df = pd.read_csv("data/insurance.csv")
        print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check required columns
        expected_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if not missing_columns:
            print("✅ All required columns present")
        else:
            print(f"❌ Missing columns: {missing_columns}")
            return False
            
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values == 0:
            print("✅ No missing values found")
        else:
            print(f"⚠️ Found {missing_values} missing values")
        
        # Show basic statistics
        print(f"📈 Dataset statistics:")
        print(f"   - Age range: {df['age'].min()}-{df['age'].max()}")
        print(f"   - BMI range: {df['bmi'].min():.1f}-{df['bmi'].max():.1f}")
        print(f"   - Charges range: ${df['charges'].min():.2f}-${df['charges'].max():.2f}")
        print(f"   - Smokers: {df['smoker'].value_counts()['yes']} ({df['smoker'].value_counts()['yes']/len(df)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\n📦 Testing package imports...")
    
    required_packages = [
        ("streamlit", "Streamlit web framework"),
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning"),
        ("seaborn", "Data visualization"),
        ("matplotlib", "Plotting"),
        ("joblib", "Model serialization")
    ]
    
    failed_imports = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (NOT INSTALLED)")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️ Missing packages: {', '.join(failed_imports)}")
        print("   Please run: pip install -r requirements.txt")
        return False
    
    return True

def test_app_structure():
    """Test if the Streamlit app has correct structure"""
    print("\n🚀 Testing Streamlit app structure...")
    
    try:
        # Read the app.py file
        with open("app/app.py", "r", encoding="utf-8") as f:
            app_content = f.read()
        
        required_functions = [
            "load_models",
            "preprocess_input", 
            "main"
        ]
        
        required_imports = [
            "import streamlit",
            "import pandas",
            "import numpy",
            "import joblib"
        ]
        
        # Check imports
        for import_stmt in required_imports:
            if import_stmt in app_content:
                print(f"✅ Found: {import_stmt}")
            else:
                print(f"❌ Missing: {import_stmt}")
        
        # Check functions
        for func in required_functions:
            if f"def {func}" in app_content:
                print(f"✅ Function defined: {func}")
            else:
                print(f"❌ Missing function: {func}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False

def main():
    """Run all tests"""
    print("🏥 Medical Cost & Insurance Approval Prediction - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Package Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("App Structure", test_app_structure)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed_tests += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "="*60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n📋 Next Steps:")
        print("1. Run the Jupyter notebook: jupyter notebook notebooks/EDA_and_Model_Training.ipynb")
        print("2. Execute all cells to train and save the models")
        print("3. Run the Streamlit app: streamlit run app/app.py")
        return True
    else:
        print("⚠️ Some tests failed. Please fix the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)