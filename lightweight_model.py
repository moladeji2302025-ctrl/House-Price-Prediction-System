"""
Lightweight model wrapper for inference without scikit-learn.
This module provides simple implementations of LinearRegression and StandardScaler
that can be used for prediction without requiring the full scikit-learn library.
"""
import numpy as np
import json
import os
import pickle


class LightweightScaler:
    """Lightweight implementation of StandardScaler for inference only."""
    
    def __init__(self, mean=None, scale=None):
        self.mean_ = np.array(mean) if mean is not None else None
        self.scale_ = np.array(scale) if scale is not None else None
    
    def transform(self, X):
        """Transform the input data using stored mean and scale."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler not initialized with mean and scale")
        
        X = np.array(X)
        return (X - self.mean_) / self.scale_
    
    def to_dict(self):
        """Convert scaler to dictionary for serialization."""
        return {
            'mean': self.mean_.tolist() if self.mean_ is not None else None,
            'scale': self.scale_.tolist() if self.scale_ is not None else None
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create scaler from dictionary."""
        return cls(mean=data['mean'], scale=data['scale'])


class LightweightLinearRegression:
    """Lightweight implementation of LinearRegression for inference only."""
    
    def __init__(self, coef=None, intercept=None):
        self.coef_ = np.array(coef) if coef is not None else None
        self.intercept_ = intercept
    
    def predict(self, X):
        """Predict using the linear model."""
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model not initialized with coefficients and intercept")
        
        X = np.array(X)
        # Handle single sample (1D array) by reshaping to 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.dot(X, self.coef_) + self.intercept_
    
    def to_dict(self):
        """Convert model to dictionary for serialization."""
        return {
            'coef': self.coef_.tolist() if self.coef_ is not None else None,
            'intercept': float(self.intercept_) if self.intercept_ is not None else None
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create model from dictionary."""
        return cls(coef=data['coef'], intercept=data['intercept'])


def save_lightweight_model(model_dict, filepath):
    """
    Save a lightweight model to JSON format.
    
    Args:
        model_dict: Dictionary containing 'model', 'scaler', and 'feature_columns'
        filepath: Path to save the JSON file
    """
    data = {
        'model': model_dict['model'].to_dict(),
        'scaler': model_dict['scaler'].to_dict(),
        'feature_columns': model_dict['feature_columns']
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_lightweight_model(filepath):
    """
    Load a lightweight model from JSON format.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Dictionary containing 'model', 'scaler', and 'feature_columns'
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return {
        'model': LightweightLinearRegression.from_dict(data['model']),
        'scaler': LightweightScaler.from_dict(data['scaler']),
        'feature_columns': data['feature_columns']
    }


def convert_sklearn_model_to_lightweight(sklearn_model_path, output_path):
    """
    Convert a scikit-learn model pickle file to lightweight JSON format.
    
    Args:
        sklearn_model_path: Path to the pickled scikit-learn model
        output_path: Path to save the lightweight JSON model
    """
    # Load the scikit-learn model
    with open(sklearn_model_path, 'rb') as f:
        sklearn_data = pickle.load(f)
    
    # Extract parameters from scikit-learn objects
    sklearn_model = sklearn_data['model']
    sklearn_scaler = sklearn_data['scaler']
    
    # Create lightweight versions
    lightweight_model = LightweightLinearRegression(
        coef=sklearn_model.coef_,
        intercept=sklearn_model.intercept_
    )
    
    lightweight_scaler = LightweightScaler(
        mean=sklearn_scaler.mean_,
        scale=sklearn_scaler.scale_
    )
    
    # Save to JSON
    save_lightweight_model({
        'model': lightweight_model,
        'scaler': lightweight_scaler,
        'feature_columns': sklearn_data['feature_columns']
    }, output_path)
    
    print(f"Converted model saved to {output_path}")


if __name__ == '__main__':
    # Convert the existing model to lightweight format
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        # Allow custom paths via command line arguments
        sklearn_path = sys.argv[2] if len(sys.argv) > 2 else 'model/house_price_model.pkl'
        output_path = sys.argv[3] if len(sys.argv) > 3 else 'model/house_price_model.json'
        
        convert_sklearn_model_to_lightweight(sklearn_path, output_path)
    else:
        print("Usage: python lightweight_model.py convert [sklearn_model_path] [output_path]")
        print("Example: python lightweight_model.py convert")
        print("Example: python lightweight_model.py convert model/my_model.pkl model/my_model.json")
