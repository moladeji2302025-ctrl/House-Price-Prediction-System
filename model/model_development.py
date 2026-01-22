"""
Model Development Script for House Price Prediction
This script trains and saves the house price prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Generate sample dataset for training."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    data = {
        'OverallQual': np.random.randint(1, 11, n_samples),
        'GrLivArea': np.random.randint(600, 5000, n_samples),
        'TotalBsmtSF': np.random.randint(0, 3000, n_samples),
        'GarageCars': np.random.randint(0, 5, n_samples),
        'GarageArea': np.random.randint(0, 1200, n_samples),
        'YearBuilt': np.random.randint(1950, 2024, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable with realistic relationships
    df['SalePrice'] = (
        50000 +
        df['OverallQual'] * 20000 +
        df['GrLivArea'] * 80 +
        df['TotalBsmtSF'] * 30 +
        df['GarageCars'] * 10000 +
        df['GarageArea'] * 50 +
        (df['YearBuilt'] - 1950) * 500 +
        np.random.normal(0, 20000, n_samples)
    )
    
    df['SalePrice'] = df['SalePrice'].clip(lower=50000)
    
    return df

def train_model():
    """Train the house price prediction model."""
    print("Creating sample dataset...")
    df = create_sample_data()
    print(f"Dataset created with {len(df)} samples")
    
    # Prepare features and target
    feature_columns = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'YearBuilt']
    X = df[feature_columns]
    y = df['SalePrice']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_test_pred = model.predict(X_test_scaled)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nMAE (Mean Absolute Error): ${test_mae:,.2f}")
    print(f"MSE (Mean Squared Error): ${test_mse:,.2f}")
    print(f"RMSE (Root Mean Squared Error): ${test_rmse:,.2f}")
    print(f"R² Score: {test_r2:.4f}")
    print("="*50)
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns
    }
    
    with open('model/house_price_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\nModel saved successfully as 'model/house_price_model.pkl'")
    
    return model, scaler, feature_columns

if __name__ == "__main__":
    train_model()
