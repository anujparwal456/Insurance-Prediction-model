# 🏥 Medical Cost & Insurance Approval Prediction

A complete end-to-end machine learning project that predicts medical insurance costs and assesses insurance approval likelihood using personal health and demographic factors.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

## 🎯 Project Objectives

This project implements two machine learning models:

1. **🔢 Regression Task**: Predict medical insurance charges using features like age, BMI, smoking status, region, etc.
2. **📊 Classification Task**: Predict insurance approval status based on cost prediction
   - ✅ **Approved (1)**: If predicted charges < ₹15,000
   - ❌ **Not Approved (0)**: If predicted charges ≥ ₹15,000

## 🗂️ Project Structure

```
Medical_Cost_Insurance_Approval/
│
├── data/
│   └── insurance.csv                    # Dataset from Kaggle
│
├── notebooks/
│   └── EDA_and_Model_Training.ipynb     # Jupyter notebook with EDA & model training
│
├── models/
│   ├── regression_model.pkl             # Trained Linear Regression model
│   ├── classification_model.pkl         # Trained Random Forest Classifier
│   ├── scaler.pkl                       # StandardScaler for feature normalization
│   └── feature_columns.pkl              # Feature column names for consistency
│
├── app/
│   └── app.py                           # Streamlit web application
│
├── requirements.txt                     # Python dependencies
└── README.md                           # Project documentation
```

## 📊 Dataset Information

**Source**: [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

**Features**:
- `age`: Age of the person
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index
- `children`: Number of children covered by health insurance
- `smoker`: Smoking status (yes/no)
- `region`: The beneficiary's residential area (northeast, northwest, southeast, southwest)
- `charges`: Individual medical costs billed by health insurance

**Dataset Statistics**:
- **Records**: 1,338 samples
- **Features**: 6 input features + 1 target variable
- **Missing Values**: None

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone or Download the Project

```bash
git clone <your-repo-url>
cd Medical_Cost_Insurance_Approval
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook (First Time Setup)

Before running the Streamlit app, you need to train and save the models:

```bash
# Navigate to notebooks directory
cd notebooks

# Start Jupyter Notebook
jupyter notebook EDA_and_Model_Training.ipynb
```

**Important**: Run all cells in the notebook to:
- Perform exploratory data analysis
- Train the regression and classification models
- Save the trained models to the `models/` directory

### 5. Run the Streamlit Application

```bash
# Navigate back to root directory
cd ..

# Run the Streamlit app
streamlit run app/app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`.

## 🚀 Usage Guide

### Using the Web Application

1. **📋 Input Patient Information**: Use the sidebar to input:
   - Age (18-100 years)
   - BMI (10.0-50.0)
   - Number of children (0-5)
   - Sex (Male/Female)
   - Smoker status (Yes/No)
   - Region (Northeast/Northwest/Southeast/Southwest)

2. **🔮 Generate Prediction**: Click the "Generate Prediction" button

3. **📊 View Results**:
   - **Cost Prediction**: Annual and monthly premium estimates
   - **Approval Status**: Likelihood of insurance approval
   - **Confidence Level**: Model prediction confidence
   - **Risk Factors**: Identified health risk factors

### Key Features

- **🎨 Interactive UI**: User-friendly Streamlit interface
- **📱 Responsive Design**: Works on desktop and mobile devices
- **🔍 Real-time Predictions**: Instant results with confidence scores
- **📊 Risk Assessment**: Automatic identification of risk factors
- **💡 Cost Breakdown**: Annual and monthly premium calculations

## 🧠 Machine Learning Models

### Regression Model (Cost Prediction)
- **Algorithm**: Linear Regression
- **Purpose**: Predict insurance premium costs
- **Features**: Age, BMI, children, sex, smoker status, region
- **Evaluation Metrics**: R², MAE, RMSE

### Classification Model (Approval Prediction)
- **Algorithm**: Random Forest Classifier
- **Purpose**: Predict insurance approval likelihood
- **Target**: Binary classification (Approved/Not Approved)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

### Data Preprocessing
- **Encoding**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler for feature normalization
- **Splitting**: 80/20 train-test split with stratification

## 📈 Model Performance

The models achieve competitive performance metrics:

- **Regression Model**: High R² score indicating good variance explanation
- **Classification Model**: High accuracy with balanced precision and recall
- **Feature Importance**: Smoking status and BMI are top predictors

*Detailed performance metrics are available in the Jupyter notebook.*

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Complete ML project"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Choose `app/app.py` as the main file
   - Click "Deploy"

3. **Access Your App**: Your app will be available at a public URL

### Option 2: Local Deployment

```bash
streamlit run app/app.py
```

### Option 3: Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py"]
```

## 📊 Dataset Citation

```
@dataset{mirichoi0218_insurance_2018,
  title={Medical Cost Personal Datasets},
  author={Miri Choi},
  year={2018},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/mirichoi0218/insurance}
}
```

## 🔒 Important Disclaimers

⚠️ **Educational Use Only**: This application is designed for educational and demonstration purposes only. 

**Key Limitations**:
- Model predictions are based on limited features and historical data patterns
- Should **not be used as the sole basis for actual insurance decisions**
- Real insurance pricing involves many additional factors not captured in this model
- Individual circumstances and company policies can significantly impact actual outcomes

**Recommendations**:
- Always consult with qualified insurance professionals for real insurance planning
- Use this tool as a learning resource for understanding ML applications in healthcare
- Verify predictions with actual insurance providers

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **🐛 Report Issues**: Found a bug? Open an issue
2. **💡 Suggest Features**: Have an idea? Let us know
3. **🔧 Submit PRs**: Improve the code or documentation
4. **📚 Improve Docs**: Help make the documentation better

### Development Setup

```bash
# Fork the repository
# Clone your fork
git clone <your-fork-url>
cd Medical_Cost_Insurance_Approval

# Create a feature branch
git checkout -b feature-name

# Make changes and commit
git commit -m "Add new feature"

# Push and create PR
git push origin feature-name
```

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🛠️ Technologies Used

- **🐍 Python 3.8+**: Core programming language
- **📊 Pandas**: Data manipulation and analysis
- **🔢 NumPy**: Numerical computing
- **🤖 Scikit-learn**: Machine learning models and preprocessing
- **📈 Matplotlib & Seaborn**: Data visualization
- **🚀 Streamlit**: Web application framework
- **💾 Joblib**: Model serialization

## 📞 Support

If you encounter any issues or have questions:

1. **📖 Check Documentation**: Review this README and the Jupyter notebook
2. **🔍 Search Issues**: Look through existing GitHub issues
3. **💬 Create Issue**: Open a new issue with detailed information
4. **📧 Contact**: Reach out via GitHub discussions

## 🌟 Acknowledgments

- **Kaggle**: For providing the insurance dataset
- **Streamlit**: For the amazing web framework
- **Scikit-learn**: For robust machine learning tools
- **Open Source Community**: For the incredible Python ecosystem

---

<div align="center">

### 🏥 Built with ❤️ for Healthcare Analytics

**Empowering informed healthcare decisions through predictive analytics**

[⭐ Star this repo](../../stargazers) | [🐛 Report Bug](../../issues) | [💡 Request Feature](../../issues)

</div>