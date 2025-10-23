# 💰 Salary Prediction using Linear Regression

A machine learning project that predicts employee salaries based on job title, location, education level, and years of experience. Developed as part of the IBM Project-Based Learning Program in AI/ML.

## 📊 Project Overview

This project implements a Linear Regression model to predict employee salaries using multiple features from a comprehensive dataset. The model provides accurate salary predictions and valuable insights into factors affecting compensation across different roles, locations, and education levels.

## 🎯 Features

- **Data Analysis**: Comprehensive EDA on salary dataset
- **Machine Learning**: Linear Regression implementation
- **Feature Engineering**: Label encoding for categorical variables
- **Model Evaluation**: R² Score, MAE, RMSE metrics
- **Visualizations**: 4 detailed charts for data insights
- **Predictive Analytics**: Salary predictions based on multiple factors

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualization

## 📁 Project Structure
```
salary-prediction/
├── salary_predictionDe.py    # Main implementation script
├── salary_dataset.csv        # Dataset (1000+ records)
├── requirements.txt          # Python dependencies
├── salary_analysis.png       # Generated visualizations
└── README.md                 # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/sahid-code404/Salary-Prediction-using-linear-regression/.git
   cd salary-prediction
```
2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Run the model**
```bash
   python salary_predictionDe.py
```

## 📊 Dataset Features

The dataset contains 1,000+ employee records with the following attributes:

| Feature | Description | Values |
|---------|-------------|--------|
| Job Title | Employee role | Software Engineer, Data Scientist, UX Designer, etc. |
| Location | Work location | San Francisco, New York, London, Bengaluru, etc. |
| Education Level | Academic qualification | High School, Bachelor's, Master's, PhD |
| YearsExperience | Professional experience | 0-12+ years |
| Salary | Target variable | $18,500 - $176,200 |

## ⚙️ Model Implementation

### Data Preprocessing

- **Label Encoding**: Convert categorical features to numerical
- **Train-Test Split**: 80-20 split with random state
- **Feature Scaling**: Not required for Linear Regression

### Machine Learning

- **Algorithm**: Linear Regression
- **Features**: Job Title, Location, Education Level, Years of Experience
- **Target**: Salary

### Evaluation Metrics

- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error

## 📈 Results & Visualizations

The script generates comprehensive visualizations:

1. **Actual vs Predicted Salary** - Scatter plot with regression line
2. **Residual Plot** - Error distribution analysis
3. **Feature Correlation** - Bar chart showing feature importance
4. **Experience vs Salary** - Relationship between experience and compensation

## 💡 Key Insights

- **Experience Matters**: Years of experience shows strongest correlation with salary
- **Education Impact**: Higher education levels significantly increase earning potential
- **Location Factor**: Geographic location plays crucial role in salary determination
- **Role Variation**: Technical roles (Data Scientist, DevOps) command higher salaries

## 🎯 Business Applications

- **HR Analytics**: Salary benchmarking and compensation planning
- **Recruitment**: Offer letter recommendations and negotiation support
- **Career Planning**: Salary expectations based on education and experience
- **Market Analysis**: Geographic and role-based salary trends

## 🤝 Contributing

This project was developed as part of IBM's Project-Based Learning Program. Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add some improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is for educational purposes as part of the IBM Project-Based Learning Program.

## 👥 Author

Developed as part of the IBM Project-Based Learning Program - AI/ML Track

---

⭐ Star this repo if you find it helpful!
