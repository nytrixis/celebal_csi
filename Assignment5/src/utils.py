"""
Utility functions for the House Price Prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets.
    
    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file
        
    Returns:
        Tuple of (train_df, test_df)
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the CSV files are in the data/ directory")
        return None, None

def get_missing_data_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get comprehensive missing data information.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing data statistics
    """
    missing_data = df.isnull().sum()
    missing_percent = 100 * missing_data / len(df)
    
    missing_table = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    
    missing_table = missing_table[missing_table['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    return missing_table

def plot_missing_data(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize missing data patterns.
    
    Args:
        df: Input DataFrame
        figsize: Figure size for the plot
    """
    missing_data = get_missing_data_info(df)
    
    if missing_data.empty:
        print("No missing data found!")
        return
    
    plt.figure(figsize=figsize)
    
    # Plot missing data percentages
    plt.subplot(2, 1, 1)
    missing_data['Missing Percentage'].plot(kind='bar')
    plt.title('Missing Data Percentage by Feature')
    plt.ylabel('Percentage Missing')
    plt.xticks(rotation=45, ha='right')
    
    # Heatmap of missing data
    plt.subplot(2, 1, 2)
    missing_cols = missing_data.index.tolist()
    sns.heatmap(df[missing_cols].isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    
    plt.tight_layout()
    plt.show()

def get_data_types_info(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize columns by data type.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with lists of column names by type
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features,
        'total_features': len(df.columns)
    }

def plot_target_distribution(target_series: pd.Series, figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot target variable distribution.
    
    Args:
        target_series: Target variable (SalePrice)
        figsize: Figure size for the plot
    """
    plt.figure(figsize=figsize)
    
    # Original distribution
    plt.subplot(1, 3, 1)
    plt.hist(target_series, bins=50, alpha=0.7, color='skyblue')
    plt.title('SalePrice Distribution')
    plt.xlabel('SalePrice')
    plt.ylabel('Frequency')
    
    # Log-transformed distribution
    plt.subplot(1, 3, 2)
    plt.hist(np.log1p(target_series), bins=50, alpha=0.7, color='lightgreen')
    plt.title('Log(SalePrice) Distribution')
    plt.xlabel('Log(SalePrice)')
    plt.ylabel('Frequency')
    
    # Box plot
    plt.subplot(1, 3, 3)
    plt.boxplot(target_series)
    plt.title('SalePrice Box Plot')
    plt.ylabel('SalePrice')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"SalePrice Statistics:")
    print(f"Mean: ${target_series.mean():,.2f}")
    print(f"Median: ${target_series.median():,.2f}")
    print(f"Std: ${target_series.std():,.2f}")
    print(f"Min: ${target_series.min():,.2f}")
    print(f"Max: ${target_series.max():,.2f}")
    print(f"Skewness: {target_series.skew():.2f}")
    print(f"Kurtosis: {target_series.kurtosis():.2f}")

def plot_correlation_matrix(df: pd.DataFrame, target_col: str = 'SalePrice', 
                          top_n: int = 20, figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot correlation matrix focusing on features most correlated with target.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        top_n: Number of top correlated features to show
        figsize: Figure size for the plot
    """
    # Calculate correlations with target
    correlations = df.corr()[target_col].abs().sort_values(ascending=False)
    
    # Get top correlated features
    top_features = correlations.head(top_n).index.tolist()
    
    # Create correlation matrix for top features
    corr_matrix = df[top_features].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title(f'Correlation Matrix - Top {top_n} Features')
    plt.tight_layout()
    plt.show()
    
    # Print top correlations
    print(f"\nTop {top_n} features correlated with {target_col}:")
    for i, (feature, corr) in enumerate(correlations.head(top_n).items(), 1):
        print(f"{i:2d}. {feature:20s}: {corr:.3f}")

def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        series: Pandas Series to check for outliers
        factor: IQR factor (typically 1.5 or 3.0)
        
    Returns:
        Boolean Series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return (series < lower_bound) | (series > upper_bound)

def print_data_summary(df: pd.DataFrame, title: str = "Data Summary") -> None:
    """
    Print comprehensive data summary.
    
    Args:
        df: Input DataFrame
        title: Title for the summary
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    data_types = get_data_types_info(df)
    print(f"\nFeature types:")
    print(f"  Numeric features: {len(data_types['numeric'])}")
    print(f"  Categorical features: {len(data_types['categorical'])}")
    print(f"  Total features: {data_types['total_features']}")
    
    # Missing data
    missing_data = get_missing_data_info(df)
    if not missing_data.empty:
        print(f"\nMissing data:")
        print(f"  Features with missing values: {len(missing_data)}")
        print(f"  Total missing values: {missing_data['Missing Count'].sum()}")
        print(f"  Percentage of total data: {missing_data['Missing Count'].sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")
    else:
        print("\nNo missing data found!")

def save_processed_data(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """
    Save processed DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        index: Whether to save index
    """
    try:
        df.to_csv(filepath, index=index)
        print(f"Data saved successfully to: {filepath}")
        print(f"Shape: {df.shape}")
    except Exception as e:
        print(f"Error saving data: {e}")
