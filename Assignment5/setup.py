"""
Setup script for House Price Prediction project.
Run this script to install dependencies and verify the setup.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages from requirements.txt."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def verify_directories():
    """Verify that all necessary directories exist."""
    print("📁 Verifying directory structure...")
    
    required_dirs = [
        "data",
        "notebooks", 
        "src",
        "outputs",
        "outputs/cleaned_data",
        "outputs/models",
        "outputs/predictions"
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}")
    
    print("✅ Directory structure verified!")

def check_data_files():
    """Check if required data files are present."""
    print("🔍 Checking for data files...")
    
    required_files = [
        "data/train.csv",
        "data/test.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            missing_files.append(file_path)
    
    if missing_files:
        print("\n⚠️  MISSING DATA FILES")
        print("Please download the following files from Kaggle:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        print("\nSee data/README.md for download instructions.")
        return False
    else:
        print("✅ All data files found!")
        return True

def verify_imports():
    """Verify that all modules can be imported."""
    print("🔧 Verifying module imports...")
    
    try:
        import pandas as pd
        print("✓ pandas")
        
        import numpy as np
        print("✓ numpy")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
        
        import seaborn as sns
        print("✓ seaborn")
        
        import sklearn
        print("✓ scikit-learn")
        
        import xgboost
        print("✓ xgboost")
        
        import lightgbm
        print("✓ lightgbm")
        
        # Test custom modules
        sys.path.append('src')
        from utils import load_data
        print("✓ utils module")
        
        from data_preprocessing import DataPreprocessor
        print("✓ data_preprocessing module")
        
        from feature_engineering import FeatureEngineer
        print("✓ feature_engineering module")
        
        from model_training import ModelTrainer
        print("✓ model_training module")
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function."""
    print("🏠 HOUSE PRICE PREDICTION PROJECT SETUP")
    print("="*50)
    
    # Step 1: Verify directories
    verify_directories()
    
    # Step 2: Install requirements
    install_success = install_requirements()
    
    if not install_success:
        print("\n❌ Setup failed during package installation.")
        return
    
    # Step 3: Verify imports
    import_success = verify_imports()
    
    if not import_success:
        print("\n❌ Setup failed during import verification.")
        return
    
    # Step 4: Check data files
    data_success = check_data_files()
    
    # Final status
    print("\n" + "="*50)
    if data_success:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("\nYou can now:")
        print("1. Run the Jupyter notebooks in order (01-04)")
        print("2. Or run the main pipeline: python main.py")
    else:
        print("⚠️  SETUP PARTIALLY COMPLETED")
        print("\nNext steps:")
        print("1. Download the required data files (see data/README.md)")
        print("2. Run this setup script again to verify")
        print("3. Then proceed with the analysis")
    
    print("\n📚 Getting Started:")
    print("• Start with notebooks/01_data_exploration.ipynb")
    print("• Follow the sequence: 01 → 02 → 03 → 04")
    print("• Check README.md for detailed instructions")

if __name__ == "__main__":
    main()
