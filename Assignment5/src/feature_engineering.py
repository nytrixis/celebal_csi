"""
Feature engineering module for House Price Prediction project.
Creates new features, encodes categorical variables, and performs feature scaling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Comprehensive feature engineering class for house price prediction.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.new_features_created = []
        self.engineering_log = []
        
    def log_action(self, action: str) -> None:
        """Log feature engineering actions."""
        self.engineering_log.append(action)
        print(f"✓ {action}")
    
    def create_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features based on domain knowledge.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        df_processed = df.copy()
        new_features = []
        
        # 1. Total living area
        if all(col in df_processed.columns for col in ['1stFlrSF', '2ndFlrSF']):
            df_processed['TotalLivingArea'] = df_processed['1stFlrSF'] + df_processed['2ndFlrSF']
            new_features.append('TotalLivingArea')
        
        # 2. Total bathrooms
        bathroom_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        if all(col in df_processed.columns for col in bathroom_cols):
            df_processed['TotalBathrooms'] = (df_processed['FullBath'] + 
                                            0.5 * df_processed['HalfBath'] + 
                                            df_processed['BsmtFullBath'] + 
                                            0.5 * df_processed['BsmtHalfBath'])
            new_features.append('TotalBathrooms')
        
        # 3. Total square footage
        if all(col in df_processed.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
            df_processed['TotalSF'] = (df_processed['TotalBsmtSF'] + 
                                     df_processed['1stFlrSF'] + 
                                     df_processed['2ndFlrSF'])
            new_features.append('TotalSF')
        
        # 4. Age of house when sold
        if all(col in df_processed.columns for col in ['YrSold', 'YearBuilt']):
            df_processed['HouseAge'] = df_processed['YrSold'] - df_processed['YearBuilt']
            new_features.append('HouseAge')
        
        # 5. Years since remodel
        if all(col in df_processed.columns for col in ['YrSold', 'YearRemodAdd']):
            df_processed['YearsSinceRemodel'] = df_processed['YrSold'] - df_processed['YearRemodAdd']
            new_features.append('YearsSinceRemodel')
        
        # 6. Total porch area
        porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        available_porch_cols = [col for col in porch_cols if col in df_processed.columns]
        if available_porch_cols:
            df_processed['TotalPorchSF'] = df_processed[available_porch_cols].sum(axis=1)
            new_features.append('TotalPorchSF')
        
        # 7. Has basement
        if 'TotalBsmtSF' in df_processed.columns:
            df_processed['HasBasement'] = (df_processed['TotalBsmtSF'] > 0).astype(int)
            new_features.append('HasBasement')
        
        # 8. Has garage
        if 'GarageArea' in df_processed.columns:
            df_processed['HasGarage'] = (df_processed['GarageArea'] > 0).astype(int)
            new_features.append('HasGarage')
        
        # 9. Has pool
        if 'PoolArea' in df_processed.columns:
            df_processed['HasPool'] = (df_processed['PoolArea'] > 0).astype(int)
            new_features.append('HasPool')
        
        # 10. Has fireplace
        if 'Fireplaces' in df_processed.columns:
            df_processed['HasFireplace'] = (df_processed['Fireplaces'] > 0).astype(int)
            new_features.append('HasFireplace')
        
        # 11. Living area per room
        if all(col in df_processed.columns for col in ['GrLivArea', 'TotRmsAbvGrd']):
            df_processed['LivAreaPerRoom'] = df_processed['GrLivArea'] / (df_processed['TotRmsAbvGrd'] + 1)  # +1 to avoid division by zero
            new_features.append('LivAreaPerRoom')
        
        # 12. Lot area per total SF
        if all(col in df_processed.columns for col in ['LotArea', 'TotalSF']):
            df_processed['LotAreaPerTotalSF'] = df_processed['LotArea'] / (df_processed['TotalSF'] + 1)
            new_features.append('LotAreaPerTotalSF')
        
        # 13. Quality-Condition interaction
        if all(col in df_processed.columns for col in ['OverallQual', 'OverallCond']):
            df_processed['QualCondInteraction'] = df_processed['OverallQual'] * df_processed['OverallCond']
            new_features.append('QualCondInteraction')
        
        # 14. Kitchen quality score (convert categorical to numerical)
        if 'KitchenQual' in df_processed.columns:
            kitchen_qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
            df_processed['KitchenQualScore'] = df_processed['KitchenQual'].map(kitchen_qual_map).fillna(0)
            new_features.append('KitchenQualScore')
        
        # 15. Exterior quality score
        if 'ExterQual' in df_processed.columns:
            exter_qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
            df_processed['ExterQualScore'] = df_processed['ExterQual'].map(exter_qual_map).fillna(0)
            new_features.append('ExterQualScore')
        
        self.new_features_created.extend(new_features)
        self.log_action(f"Created {len(new_features)} new features: {', '.join(new_features)}")
        
        return df_processed
    
    def encode_categorical_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None,
                                   encoding_method: str = 'onehot', max_categories: int = 10) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Encode categorical features using specified method.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            encoding_method: 'onehot', 'label', or 'target'
            max_categories: Maximum categories for one-hot encoding
            
        Returns:
            Tuple of (encoded_train_df, encoded_test_df)
        """
        train_encoded = train_df.copy()
        test_encoded = test_df.copy() if test_df is not None else None
        
        # Get categorical columns
        categorical_cols = train_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if encoding_method == 'onehot':
            # One-hot encoding for low cardinality features
            for col in categorical_cols:
                n_categories = train_encoded[col].nunique()
                
                if n_categories <= max_categories:
                    # One-hot encode
                    train_dummies = pd.get_dummies(train_encoded[col], prefix=col, drop_first=True)
                    train_encoded = pd.concat([train_encoded, train_dummies], axis=1)
                    train_encoded.drop(columns=[col], inplace=True)
                    
                    if test_encoded is not None:
                        test_dummies = pd.get_dummies(test_encoded[col], prefix=col, drop_first=True)
                        # Align columns with training data
                        for dummy_col in train_dummies.columns:
                            if dummy_col not in test_dummies.columns:
                                test_dummies[dummy_col] = 0
                        test_dummies = test_dummies[train_dummies.columns]
                        test_encoded = pd.concat([test_encoded, test_dummies], axis=1)
                        test_encoded.drop(columns=[col], inplace=True)
                else:
                    # Label encode for high cardinality features
                    le = LabelEncoder()
                    train_encoded[col] = le.fit_transform(train_encoded[col].astype(str))
                    self.encoders[col] = le
                    
                    if test_encoded is not None:
                        # Handle unseen categories in test data
                        test_encoded[col] = test_encoded[col].astype(str)
                        test_encoded[col] = test_encoded[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        elif encoding_method == 'label':
            # Label encoding for all categorical features
            for col in categorical_cols:
                le = LabelEncoder()
                train_encoded[col] = le.fit_transform(train_encoded[col].astype(str))
                self.encoders[col] = le
                
                if test_encoded is not None:
                    test_encoded[col] = test_encoded[col].astype(str)
                    test_encoded[col] = test_encoded[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        self.log_action(f"Categorical features encoded using {encoding_method} method. "
                       f"Processed {len(categorical_cols)} categorical features.")
        
        return train_encoded, test_encoded
    
    def create_polynomial_features(self, df: pd.DataFrame, features: List[str], degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.
        
        Args:
            df: Input DataFrame
            features: List of feature names to create polynomials for
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        df_processed = df.copy()
        new_poly_features = []
        
        for feature in features:
            if feature in df_processed.columns:
                for d in range(2, degree + 1):
                    poly_feature_name = f"{feature}_poly_{d}"
                    df_processed[poly_feature_name] = df_processed[feature] ** d
                    new_poly_features.append(poly_feature_name)
        
        if new_poly_features:
            self.log_action(f"Created {len(new_poly_features)} polynomial features of degree {degree}")
        
        return df_processed
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples containing feature pairs
            
        Returns:
            DataFrame with interaction features
        """
        df_processed = df.copy()
        new_interaction_features = []
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df_processed.columns and feat2 in df_processed.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                df_processed[interaction_name] = df_processed[feat1] * df_processed[feat2]
                new_interaction_features.append(interaction_name)
        
        if new_interaction_features:
            self.log_action(f"Created {len(new_interaction_features)} interaction features")
        
        return df_processed
    
    def scale_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None,
                      method: str = 'standard', exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Scale numerical features.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            method: Scaling method ('standard', 'minmax', 'robust')
            exclude_cols: Columns to exclude from scaling
            
        Returns:
            Tuple of (scaled_train_df, scaled_test_df)
        """
        train_scaled = train_df.copy()
        test_scaled = test_df.copy() if test_df is not None else None
        
        # Get numerical columns
        numerical_cols = train_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        if exclude_cols:
            numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Initialize scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        # Fit and transform training data
        train_scaled[numerical_cols] = scaler.fit_transform(train_scaled[numerical_cols])
        self.scalers[method] = scaler
        
        # Transform test data
        if test_scaled is not None:
            test_scaled[numerical_cols] = scaler.transform(test_scaled[numerical_cols])
        
        self.log_action(f"Features scaled using {method} method. Scaled {len(numerical_cols)} numerical features.")
        
        return train_scaled, test_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'f_regression',
                       k: int = 50) -> pd.DataFrame:
        """
        Select top k features using specified method.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Feature selection method ('f_regression', 'rfe')
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")
        
        self.feature_selectors[method] = selector
        self.log_action(f"Selected {len(selected_features)} features using {method} method")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def feature_engineering_pipeline(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None,
                                   target_col: str = 'SalePrice', 
                                   encoding_method: str = 'onehot',
                                   scaling_method: str = 'standard',
                                   create_polynomials: bool = True,
                                   create_interactions: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Complete feature engineering pipeline.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            target_col: Target column name
            encoding_method: Categorical encoding method
            scaling_method: Feature scaling method
            create_polynomials: Whether to create polynomial features
            create_interactions: Whether to create interaction features
            
        Returns:
            Tuple of (engineered_train_df, engineered_test_df)
        """
        print("Starting feature engineering pipeline...")
        print("="*50)
        
        # Separate target variable
        if target_col in train_df.columns:
            y_train = train_df[target_col].copy()
            X_train = train_df.drop(columns=[target_col])
        else:
            y_train = None
            X_train = train_df.copy()
        
        X_test = test_df.copy() if test_df is not None else None
        
        # 1. Create new features
        X_train = self.create_new_features(X_train)
        if X_test is not None:
            X_test = self.create_new_features(X_test)
        
        # 2. Create polynomial features for key numerical features
        if create_polynomials:
            key_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']
            available_features = [f for f in key_features if f in X_train.columns]
            if available_features:
                X_train = self.create_polynomial_features(X_train, available_features, degree=2)
                if X_test is not None:
                    X_test = self.create_polynomial_features(X_test, available_features, degree=2)
        
        # 3. Create interaction features
        if create_interactions:
            important_pairs = [
                ('OverallQual', 'GrLivArea'),
                ('TotalBsmtSF', 'GrLivArea'),
                ('GarageArea', 'GrLivArea'),
                ('OverallQual', 'TotalBsmtSF')
            ]
            available_pairs = [(f1, f2) for f1, f2 in important_pairs 
                             if f1 in X_train.columns and f2 in X_train.columns]
            if available_pairs:
                X_train = self.create_interaction_features(X_train, available_pairs)
                if X_test is not None:
                    X_test = self.create_interaction_features(X_test, available_pairs)
        
        # 4. Encode categorical features
        X_train, X_test = self.encode_categorical_features(X_train, X_test, encoding_method)
        
        # 5. Scale features (exclude target and ID columns)
        exclude_from_scaling = ['Id']
        if y_train is not None:
            exclude_from_scaling.append(target_col)
        
        X_train, X_test = self.scale_features(X_train, X_test, scaling_method, exclude_from_scaling)
        
        # Add target back to training data
        if y_train is not None:
            X_train[target_col] = y_train
        
        # Log final results
        print("\n" + "="*50)
        print("FEATURE ENGINEERING COMPLETED")
        print("="*50)
        print(f"Training data shape: {X_train.shape}")
        if X_test is not None:
            print(f"Test data shape: {X_test.shape}")
        print(f"New features created: {len(self.new_features_created)}")
        
        print(f"\nFeature engineering steps completed:")
        for i, action in enumerate(self.engineering_log, 1):
            print(f"{i}. {action}")
        
        return X_train, X_test
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series, n_features: int = 20) -> pd.DataFrame:
        """
        Get feature importance using Random Forest.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        # Use only numerical features for this analysis
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_features]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_numerical, y)
        
        importance_df = pd.DataFrame({
            'feature': X_numerical.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(n_features)
