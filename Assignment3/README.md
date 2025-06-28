# Iris Dataset Visualization

## Description
This project explores the Iris dataset using Python and Jupyter Notebook.  
It includes data visualizations such as histograms, boxplots, pairplots, and heatmaps.

## How to Run
1. Install requirements:  
   `pip install pandas matplotlib seaborn`
2. Open `iris_eda.ipynb` in Jupyter Notebook or VS Code.
3. Run all cells.

## Summary
- Visualized feature distributions, outliers, and relationships between variables and species.

# Titanic Dataset - In-depth Exploratory Data Analysis

## Overview
This project conducts a comprehensive EDA on the Titanic dataset, focusing on understanding data distributions, identifying missing values, detecting outliers, and uncovering relationships between variables.

## Dataset
The Titanic dataset contains information about passengers aboard the RMS Titanic, including demographics, ticket information, and survival status.

## Key Analysis Areas
1. **Data Structure**: Understanding dataset dimensions and data types
2. **Missing Values**: Identifying and visualizing missing data patterns
3. **Data Distributions**: Analyzing numerical and categorical variable distributions
4. **Outlier Detection**: Using box plots and statistical methods to identify outliers
5. **Correlation Analysis**: Understanding relationships between numerical variables
6. **Survival Analysis**: Exploring factors that influenced passenger survival

## Visualizations Used
- Histograms for distribution analysis
- Box plots for outlier detection
- Heatmaps for correlation and missing value analysis
- Count plots for categorical variables
- Violin plots for distribution comparison
- Pairplots for multi-variable relationships

## How to Run
1. Install required packages: `pip install pandas numpy matplotlib seaborn`
2. Download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data)
3. Place `titanic.csv` in the project directory
4. Run `titanic_eda_complete.ipynb` in Jupyter Notebook or VS Code

## Key Findings
- Significant missing values in Age (19.9%) and Cabin (77.1%) columns
- Strong correlation between passenger class and survival rate
- Gender was a major factor in survival (74.2% female vs 18.9% male survival rate)
- First-class passengers had higher survival rates than other classes
- Age distribution shows slight difference between survivors and non-survivors

## Resources
- [Kaggle Titanic EDA Reference](https://www.kaggle.com/code/junaiddata35/titanic-dataset-exploratory-data-analysis-eda)