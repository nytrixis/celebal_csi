# House Price Prediction Project

## Overview
This project implements a machine learning solution for predicting house prices using the Kaggle House Prices dataset. The project focuses on comprehensive data preprocessing and feature engineering techniques suitable for entry-level data science practitioners.

## Dataset
- **Source**: [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- **Target Variable**: SalePrice (house sale price in dollars)
- **Features**: 79 explanatory variables describing residential properties in Ames, Iowa

## Project Structure
```
Assignment5/
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv (download from Kaggle)
│   ├── test.csv (download from Kaggle)
│   └── data_description.txt (download from Kaggle)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
└── outputs/
    ├── cleaned_data/
    ├── models/
    └── predictions/
```

## Key Features
1. **Data Exploration**: Comprehensive analysis of data distribution, missing values, and correlations
2. **Data Preprocessing**: Handling missing values, outliers, and data type conversions
3. **Feature Engineering**: Creating new features, encoding categorical variables, and feature scaling
4. **Model Training**: Multiple regression models with hyperparameter tuning
5. **Model Evaluation**: Cross-validation and performance metrics

## Getting Started
1. Clone/download the project
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset files from Kaggle and place in `data/` folder
4. Run the Jupyter notebooks in order (01-04)

## Skills Demonstrated
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering and selection
- Machine learning model implementation
- Model evaluation and validation
