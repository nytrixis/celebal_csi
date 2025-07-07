"""
Main script demonstrating the complete house price prediction pipeline.
This script shows how to use all the modules together.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.utils import *
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer

def main():
    """
    Main function to run the complete pipeline.
    """
    print("🏠 HOUSE PRICE PREDICTION PIPELINE")
    print("="*50)
    
    # 1. Load Data
    print("\n1️⃣ LOADING DATA...")
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    try:
        train_df, test_df = load_data(train_path, test_path)
        if train_df is None or test_df is None:
            print("❌ Failed to load data. Please ensure CSV files are in the data/ directory.")
            return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Please download the dataset from Kaggle and place in data/ directory.")
        return
    
    # 2. Data Preprocessing
    print("\n2️⃣ DATA PREPROCESSING...")
    preprocessor = DataPreprocessor()
    train_processed, test_processed = preprocessor.preprocess_pipeline(
        train_df, test_df, target_col='SalePrice'
    )
    
    # 3. Feature Engineering
    print("\n3️⃣ FEATURE ENGINEERING...")
    feature_engineer = FeatureEngineer()
    train_engineered, test_engineered = feature_engineer.feature_engineering_pipeline(
        train_processed, test_processed, target_col='SalePrice'
    )
    
    # Separate features and target
    X_train = train_engineered.drop('SalePrice', axis=1)
    y_train = train_engineered['SalePrice']
    X_test = test_engineered
    
    # 4. Model Training
    print("\n4️⃣ MODEL TRAINING...")
    trainer = ModelTrainer(random_state=42)
    
    # Train baseline models
    print("Training baseline models...")
    baseline_results = trainer.train_baseline_models(X_train, y_train)
    print(f"Best baseline model: {baseline_results.iloc[0]['Model']}")
    print(f"CV RMSE: {baseline_results.iloc[0]['CV_RMSE_Mean']:.4f}")
    
    # Hyperparameter tuning for top 3 models
    print("\nTuning hyperparameters for top models...")
    top_models = baseline_results.head(3)['Model'].tolist()
    best_models = trainer.tune_hyperparameters(
        X_train, y_train, 
        model_names=top_models,
        search_method='random',
        n_iter=20  # Reduced for demo
    )
    
    # 5. Generate Predictions
    print("\n5️⃣ GENERATING PREDICTIONS...")
    if best_models:
        # Get the best model
        best_model_name = list(best_models.keys())[0]
        best_model = best_models[best_model_name]
        
        # Generate predictions
        predictions = trainer.generate_predictions(best_model, X_test)
        
        if predictions is not None:
            # Create submission file
            submission = pd.DataFrame({
                'Id': test_df['Id'],
                'SalePrice': predictions
            })
            
            # Save predictions
            output_path = 'outputs/predictions/submission.csv'
            submission.to_csv(output_path, index=False)
            print(f"✅ Predictions saved to: {output_path}")
            
            # Show prediction statistics
            print(f"\n📊 PREDICTION STATISTICS:")
            print(f"Predicted price range: ${predictions.min():,.0f} - ${predictions.max():,.0f}")
            print(f"Median predicted price: ${np.median(predictions):,.0f}")
            
    # 6. Save Models and Data
    print("\n6️⃣ SAVING OUTPUTS...")
    
    # Save preprocessed data
    train_processed.to_csv('outputs/cleaned_data/train_preprocessed.csv', index=False)
    test_processed.to_csv('outputs/cleaned_data/test_preprocessed.csv', index=False)
    
    # Save engineered data
    train_engineered.to_csv('outputs/cleaned_data/train_engineered.csv', index=False)
    test_engineered.to_csv('outputs/cleaned_data/test_engineered.csv', index=False)
    
    # Save best model
    if best_models:
        trainer.save_best_model(best_model_name, 'outputs/models/best_model.pkl')
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Next steps:")
    print("1. Review the Jupyter notebooks for detailed analysis")
    print("2. Experiment with different feature engineering techniques")
    print("3. Try ensemble methods for better performance")
    print("4. Submit predictions to Kaggle competition")

if __name__ == "__main__":
    main()
