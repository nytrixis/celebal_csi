"""
Model training module for House Price Prediction project.
Implements multiple regression models with hyperparameter tuning and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine learning models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

# Model evaluation and selection
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """
    Comprehensive model training class for house price prediction.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.evaluation_results = {}
        self.training_log = []
        
    def log_action(self, action: str) -> None:
        """Log training actions."""
        self.training_log.append(action)
        print(f"✓ {action}")
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize different regression models with default parameters.
        
        Returns:
            Dictionary of model instances
        """
        models = {
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(random_state=self.random_state),
            
            'Lasso Regression': Lasso(random_state=self.random_state),
            
            'Elastic Net': ElasticNet(random_state=self.random_state),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            ),
            
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.random_state,
                verbosity=0
            ),
            
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                random_state=self.random_state,
                verbosity=-1
            ),
            
            'Support Vector Regression': SVR(),
            
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=500
            )
        }
        
        self.models = models
        self.log_action(f"Initialized {len(models)} regression models")
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """
        Get hyperparameter grids for model tuning.
        
        Returns:
            Dictionary of hyperparameter grids for each model
        """
        param_grids = {
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            },
            
            'Lasso Regression': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            
            'Elastic Net': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            
            'Support Vector Regression': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            },
            
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        }
        
        return param_grids
    
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, float]:
        """
        Evaluate model using cross-validation.
        
        Args:
            model: Trained model instance
            X: Feature DataFrame
            y: Target Series
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
        
        # Convert negative MSE to positive RMSE
        if scoring == 'neg_mean_squared_error':
            rmse_scores = np.sqrt(-cv_scores)
            mean_rmse = rmse_scores.mean()
            std_rmse = rmse_scores.std()
        
        # Fit model to get additional metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        evaluation_metrics = {
            'CV_RMSE_Mean': mean_rmse,
            'CV_RMSE_Std': std_rmse,
            'Train_RMSE': rmse,
            'Train_MAE': mae,
            'Train_R2': r2,
            'Train_MSE': mse
        }
        
        return evaluation_metrics
    
    def train_baseline_models(self, X: pd.DataFrame, y: pd.Series, 
                            cv_folds: int = 5) -> pd.DataFrame:
        """
        Train baseline models without hyperparameter tuning.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv_folds: Number of cross-validation folds
            
        Returns:
            DataFrame with baseline model results
        """
        if not self.models:
            self.initialize_models()
        
        results = []
        
        print("Training baseline models...")
        print("-" * 50)
        
        for name, model in self.models.items():
            try:
                print(f"Training {name}...")
                metrics = self.evaluate_model(model, X, y, cv_folds)
                
                result = {'Model': name}
                result.update(metrics)
                results.append(result)
                
                print(f"  CV RMSE: {metrics['CV_RMSE_Mean']:.4f} (+/- {metrics['CV_RMSE_Std']:.4f})")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('CV_RMSE_Mean')
        
        self.evaluation_results['baseline'] = results_df
        self.log_action(f"Baseline models trained. Best model: {results_df.iloc[0]['Model']}")
        
        return results_df
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                           model_names: List[str] = None,
                           search_method: str = 'grid',
                           cv_folds: int = 5,
                           n_iter: int = 50) -> Dict[str, Any]:
        """
        Tune hyperparameters for specified models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_names: List of model names to tune (None for all)
            search_method: 'grid' or 'random'
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for random search
            
        Returns:
            Dictionary of best models
        """
        if not self.models:
            self.initialize_models()
        
        param_grids = self.get_hyperparameter_grids()
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        best_models = {}
        tuning_results = []
        
        print(f"Hyperparameter tuning using {search_method} search...")
        print("-" * 50)
        
        for name in model_names:
            if name not in self.models or name not in param_grids:
                print(f"Skipping {name} - not available or no param grid")
                continue
            
            try:
                print(f"Tuning {name}...")
                
                model = self.models[name]
                param_grid = param_grids[name]
                
                # Choose search method
                if search_method == 'grid':
                    search = GridSearchCV(
                        model, param_grid, cv=cv_folds,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1, verbose=0
                    )
                else:  # random search
                    search = RandomizedSearchCV(
                        model, param_grid, cv=cv_folds,
                        scoring='neg_mean_squared_error',
                        n_iter=n_iter, n_jobs=-1,
                        random_state=self.random_state, verbose=0
                    )
                
                # Fit search
                search.fit(X, y)
                
                # Store best model
                best_models[name] = search.best_estimator_
                
                # Evaluate best model
                best_score = np.sqrt(-search.best_score_)
                
                tuning_results.append({
                    'Model': name,
                    'Best_CV_RMSE': best_score,
                    'Best_Params': search.best_params_
                })
                
                print(f"  Best CV RMSE: {best_score:.4f}")
                print(f"  Best params: {search.best_params_}")
                
            except Exception as e:
                print(f"  Error tuning {name}: {str(e)}")
                continue
        
        self.best_models = best_models
        tuning_df = pd.DataFrame(tuning_results)
        if not tuning_df.empty:
            tuning_df = tuning_df.sort_values('Best_CV_RMSE')
            self.evaluation_results['tuned'] = tuning_df
            self.log_action(f"Hyperparameter tuning completed. Best model: {tuning_df.iloc[0]['Model']}")
        
        return best_models
    
    def create_ensemble_model(self, X: pd.DataFrame, y: pd.Series, 
                            model_names: List[str] = None) -> Any:
        """
        Create ensemble model using weighted average of best models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_names: List of model names to include in ensemble
            
        Returns:
            Ensemble model (as a custom class)
        """
        if not self.best_models:
            print("No tuned models available. Please run hyperparameter tuning first.")
            return None
        
        if model_names is None:
            # Use top 3 models
            if 'tuned' in self.evaluation_results:
                top_models = self.evaluation_results['tuned'].head(3)['Model'].tolist()
                model_names = [name for name in top_models if name in self.best_models]
            else:
                model_names = list(self.best_models.keys())[:3]
        
        # Filter available models
        available_models = {name: model for name, model in self.best_models.items() 
                          if name in model_names}
        
        if len(available_models) < 2:
            print("Need at least 2 models for ensemble")
            return None
        
        # Create ensemble
        ensemble = EnsembleModel(available_models)
        ensemble.fit(X, y)
        
        self.log_action(f"Ensemble model created with {len(available_models)} models")
        
        return ensemble
    
    def plot_model_comparison(self, results_df: pd.DataFrame, metric: str = 'CV_RMSE_Mean',
                            figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot model comparison results.
        
        Args:
            results_df: DataFrame with model results
            metric: Metric to plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create bar plot
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(results_df)), results_df[metric])
        plt.xticks(range(len(results_df)), results_df['Model'], rotation=45, ha='right')
        plt.ylabel(metric)
        plt.title(f'Model Comparison - {metric}')
        
        # Color bars (best model in green)
        bars[0].set_color('green')
        for i in range(1, len(bars)):
            bars[i].set_color('skyblue')
        
        # Create horizontal bar plot for better readability
        plt.subplot(2, 1, 2)
        plt.barh(range(len(results_df)), results_df[metric])
        plt.yticks(range(len(results_df)), results_df['Model'])
        plt.xlabel(metric)
        plt.title(f'Model Comparison - {metric} (Horizontal)')
        
        # Add value labels
        for i, v in enumerate(results_df[metric]):
            plt.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def save_best_model(self, model_name: str, filepath: str) -> None:
        """
        Save the best model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name in self.best_models:
            joblib.dump(self.best_models[model_name], filepath)
            self.log_action(f"Model {model_name} saved to {filepath}")
        else:
            print(f"Model {model_name} not found in best models")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(filepath)
            self.log_action(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def generate_predictions(self, model: Any, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            
        Returns:
            Predictions array
        """
        try:
            predictions = model.predict(X_test)
            self.log_action(f"Generated {len(predictions)} predictions")
            return predictions
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return None

class EnsembleModel:
    """
    Simple ensemble model using weighted average.
    """
    
    def __init__(self, models: Dict[str, Any], weights: List[float] = None):
        self.models = models
        self.model_names = list(models.keys())
        self.weights = weights or [1.0] * len(models)  # Equal weights by default
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """Fit all models in the ensemble."""
        for name, model in self.models.items():
            model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble model must be fitted before making predictions")
        
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(self.weights) / np.sum(self.weights)  # Normalize weights
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def get_model_names(self) -> List[str]:
        """Get names of models in ensemble."""
        return self.model_names
