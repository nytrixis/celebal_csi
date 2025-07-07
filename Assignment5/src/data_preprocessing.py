"""
Data preprocessing module for House Price Prediction project.
Handles missing values, outliers, and data cleaning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing class for house price prediction.
    """
    
    def __init__(self):
        self.numerical_imputers = {}
        self.categorical_imputers = {}
        self.outlier_bounds = {}
        self.preprocessing_log = []
        
    def log_action(self, action: str) -> None:
        """Log preprocessing actions."""
        self.preprocessing_log.append(action)
        print(f"✓ {action}")
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values based on feature type and missingness pattern.
        
        Args:
            df: Input DataFrame
            strategy: Custom strategy dict {column_name: strategy}
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        # Default strategies for known features
        default_strategies = {
            # Basement features - NA likely means "No Basement"
            'BsmtQual': 'None',
            'BsmtCond': 'None', 
            'BsmtExposure': 'None',
            'BsmtFinType1': 'None',
            'BsmtFinType2': 'None',
            'BsmtFinSF1': 0,
            'BsmtFinSF2': 0,
            'BsmtUnfSF': 0,
            'TotalBsmtSF': 0,
            'BsmtFullBath': 0,
            'BsmtHalfBath': 0,
            
            # Garage features - NA likely means "No Garage"
            'GarageType': 'None',
            'GarageFinish': 'None',
            'GarageQual': 'None',
            'GarageCond': 'None',
            'GarageYrBlt': 0,
            'GarageArea': 0,
            'GarageCars': 0,
            
            # Pool features - NA likely means "No Pool"
            'PoolQC': 'None',
            'PoolArea': 0,
            
            # Other features
            'Fence': 'None',
            'Alley': 'None',
            'MiscFeature': 'None',
            'FireplaceQu': 'None',
            
            # Masonry veneer
            'MasVnrType': 'None',
            'MasVnrArea': 0,
            
            # Lot frontage - use median by neighborhood
            'LotFrontage': 'neighborhood_median'
        }
        
        # Merge with custom strategies
        if strategy:
            default_strategies.update(strategy)
        
        for column in df_processed.columns:
            if df_processed[column].isnull().sum() > 0:
                
                if column in default_strategies:
                    strategy_type = default_strategies[column]
                    
                    if strategy_type == 'neighborhood_median' and column == 'LotFrontage':
                        # Special handling for LotFrontage
                        if 'Neighborhood' in df_processed.columns:
                            df_processed['LotFrontage'] = df_processed.groupby('Neighborhood')['LotFrontage'].transform(
                                lambda x: x.fillna(x.median())
                            )
                            # Fill any remaining NAs with overall median
                            df_processed['LotFrontage'].fillna(df_processed['LotFrontage'].median(), inplace=True)
                        else:
                            df_processed[column].fillna(df_processed[column].median(), inplace=True)
                    
                    elif isinstance(strategy_type, (int, float)):
                        df_processed[column].fillna(strategy_type, inplace=True)
                    
                    elif isinstance(strategy_type, str):
                        df_processed[column].fillna(strategy_type, inplace=True)
                
                else:
                    # Default strategies for remaining columns
                    if df_processed[column].dtype in ['object']:
                        # Categorical: mode or 'Unknown'
                        mode_value = df_processed[column].mode()
                        fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                        df_processed[column].fillna(fill_value, inplace=True)
                    else:
                        # Numerical: median
                        df_processed[column].fillna(df_processed[column].median(), inplace=True)
        
        missing_after = df_processed.isnull().sum().sum()
        self.log_action(f"Missing values handled. Remaining missing values: {missing_after}")
        
        return df_processed
    
    def handle_outliers(self, df: pd.DataFrame, target_col: str = None, 
                       method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in numerical features.
        
        Args:
            df: Input DataFrame
            target_col: Target column name (if present)
            method: Outlier detection method ('iqr', 'zscore')
            factor: Factor for outlier detection (1.5 for IQR, 3 for Z-score)
            
        Returns:
            DataFrame with outliers handled
        """
        df_processed = df.copy()
        outliers_removed = 0
        
        # Get numerical columns (excluding target if present)
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        # Columns to check for outliers (excluding ID columns and some specific features)
        exclude_cols = ['Id', 'MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 
                       'MoSold', 'YrSold', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']
        
        cols_to_check = [col for col in numerical_cols if col not in exclude_cols]
        
        for column in cols_to_check:
            if method == 'iqr':
                Q1 = df_processed[column].quantile(0.25)
                Q3 = df_processed[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Store bounds for future use
                self.outlier_bounds[column] = (lower_bound, upper_bound)
                
                # Cap outliers instead of removing them
                outlier_count = ((df_processed[column] < lower_bound) | 
                               (df_processed[column] > upper_bound)).sum()
                
                if outlier_count > 0:
                    df_processed[column] = np.clip(df_processed[column], lower_bound, upper_bound)
                    outliers_removed += outlier_count
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_processed[column]))
                outlier_mask = z_scores > factor
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    # Cap outliers at mean ± factor*std
                    mean_val = df_processed[column].mean()
                    std_val = df_processed[column].std()
                    lower_bound = mean_val - factor * std_val
                    upper_bound = mean_val + factor * std_val
                    
                    df_processed[column] = np.clip(df_processed[column], lower_bound, upper_bound)
                    outliers_removed += outlier_count
        
        self.log_action(f"Outliers handled using {method} method. {outliers_removed} outliers capped.")
        
        return df_processed
    
    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix data types for specific columns based on domain knowledge.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with corrected data types
        """
        df_processed = df.copy()
        
        # Columns that should be categorical (string) but might be numerical
        categorical_cols = {
            'MSSubClass': 'category',  # Building class
            'OverallQual': 'int',      # Keep as int for easier processing
            'OverallCond': 'int',      # Keep as int for easier processing
            'YrSold': 'int',           # Year sold
            'MoSold': 'int',           # Month sold
        }
        
        # Columns that should be numerical but might be object
        numerical_cols = {
            'LotFrontage': 'float64',
            'LotArea': 'int',
            'MasVnrArea': 'float64',
        }
        
        # Apply categorical conversions
        for col, dtype in categorical_cols.items():
            if col in df_processed.columns:
                try:
                    if dtype == 'category':
                        df_processed[col] = df_processed[col].astype('category')
                    else:
                        df_processed[col] = df_processed[col].astype(dtype)
                except:
                    pass  # Skip if conversion fails
        
        # Apply numerical conversions
        for col, dtype in numerical_cols.items():
            if col in df_processed.columns:
                try:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    df_processed[col] = df_processed[col].astype(dtype)
                except:
                    pass  # Skip if conversion fails
        
        self.log_action("Data types corrected based on domain knowledge.")
        
        return df_processed
    
    def handle_rare_categories(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Handle rare categories by grouping them into 'Other'.
        
        Args:
            df: Input DataFrame
            threshold: Minimum frequency threshold (as percentage)
            
        Returns:
            DataFrame with rare categories handled
        """
        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        
        changes_made = 0
        
        for col in categorical_cols:
            if col in df_processed.columns:
                # Calculate value counts and frequencies
                value_counts = df_processed[col].value_counts()
                frequencies = value_counts / len(df_processed)
                
                # Find rare categories
                rare_categories = frequencies[frequencies < threshold].index.tolist()
                
                if len(rare_categories) > 0:
                    df_processed[col] = df_processed[col].replace(rare_categories, 'Other')
                    changes_made += len(rare_categories)
        
        if changes_made > 0:
            self.log_action(f"Rare categories handled. {changes_made} categories grouped into 'Other'.")
        
        return df_processed
    
    def preprocess_pipeline(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None,
                           target_col: str = 'SalePrice') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            target_col: Target column name
            
        Returns:
            Tuple of (processed_train_df, processed_test_df)
        """
        print("Starting preprocessing pipeline...")
        print("="*50)
        
        # Store original shapes
        train_original_shape = train_df.shape
        test_original_shape = test_df.shape if test_df is not None else None
        
        # Separate target variable
        if target_col in train_df.columns:
            y_train = train_df[target_col].copy()
            X_train = train_df.drop(columns=[target_col])
        else:
            y_train = None
            X_train = train_df.copy()
        
        # Process training data
        X_train_processed = self.fix_data_types(X_train)
        X_train_processed = self.handle_missing_values(X_train_processed)
        X_train_processed = self.handle_outliers(X_train_processed)
        X_train_processed = self.handle_rare_categories(X_train_processed)
        
        # Add target back to training data
        if y_train is not None:
            train_processed = X_train_processed.copy()
            train_processed[target_col] = y_train
        else:
            train_processed = X_train_processed
        
        # Process test data if provided
        test_processed = None
        if test_df is not None:
            test_processed = self.fix_data_types(test_df)
            test_processed = self.handle_missing_values(test_processed)
            # Note: Don't handle outliers in test data based on test data statistics
            # Use training data bounds if needed
            test_processed = self.handle_rare_categories(test_processed)
        
        # Log final results
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETED")
        print("="*50)
        print(f"Training data: {train_original_shape} → {train_processed.shape}")
        if test_processed is not None:
            print(f"Test data: {test_original_shape} → {test_processed.shape}")
        
        print(f"\nPreprocessing steps completed:")
        for i, action in enumerate(self.preprocessing_log, 1):
            print(f"{i}. {action}")
        
        return train_processed, test_processed
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of preprocessing steps performed.
        
        Returns:
            Dictionary with preprocessing summary
        """
        return {
            'steps_completed': len(self.preprocessing_log),
            'preprocessing_log': self.preprocessing_log,
            'outlier_bounds': self.outlier_bounds,
            'imputers_fitted': {
                'numerical': list(self.numerical_imputers.keys()),
                'categorical': list(self.categorical_imputers.keys())
            }
        }
