# House Price Prediction System

A machine learning-based web application that predicts house prices using Linear Regression. The system analyzes six key features of a house to provide accurate price estimates.

## 🎯 Features

- **Machine Learning Model**: Linear Regression trained on house features
- **Web Interface**: User-friendly Flask-based web application
- **Real-time Predictions**: Instant price estimates based on user input
- **6 Key Features**:
  - Overall Quality (1-10)
  - Living Area (sq ft)
  - Basement Area (sq ft)
  - Garage Capacity (number of cars)
  - Garage Area (sq ft)
  - Year Built

## 📁 Project Structure

```
House-Price-Prediction-System/
├── model/
│   ├── model_building.ipynb       # Jupyter notebook for model development
│   ├── model_development.py       # Python script for model training
│   ├── house_price_model.pkl      # Trained model in scikit-learn format (for training)
│   └── house_price_model.json     # Lightweight model for deployment (generated)
├── templates/
│   └── index.html                 # Web interface
├── static/                        # (Optional) CSS/JS files
├── app.py                         # Flask web application
├── lightweight_model.py           # Lightweight model implementation (no scikit-learn)
├── requirements.txt               # Runtime dependencies (minimal for deployment)
├── requirements-dev.txt           # Development and training dependencies
├── vercel.json                    # Vercel deployment configuration
├── .vercelignore                  # Files to exclude from deployment
├── HousePrice_hosted_webGUI_link.txt  # Deployment details
└── README.md                      # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/moladeji2302025-ctrl/House-Price-Prediction-System.git
   cd House-Price-Prediction-System
   ```

2. **Install dependencies**
   
   For development and training:
   ```bash
   pip install -r requirements-dev.txt
   ```
   
   For production/deployment only:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if not already trained)
   ```bash
   python model/model_development.py
   ```
   This will generate `model/house_price_model.pkl`

4. **Convert model to lightweight format** (for deployment)
   ```bash
   python lightweight_model.py convert
   ```
   This will generate `model/house_price_model.json` - a lightweight version without scikit-learn dependencies

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the web interface**
   Open your browser and navigate to: `http://localhost:5000`

## 📊 Model Development

The model is developed using the following steps:

1. **Data Preprocessing**
   - Handle missing values
   - Scale numeric features using StandardScaler
   - Feature engineering

2. **Model Training**
   - Algorithm: Linear Regression
   - Features: 6 carefully selected features
   - Train-test split: 80-20

3. **Model Evaluation**
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - R² Score

4. **Model Persistence**
   - Development: Saved using Python's pickle module (scikit-learn format)
   - Production: Converted to lightweight JSON format
   - Includes model coefficients, scaler parameters, and feature columns
   - JSON format enables deployment without scikit-learn dependencies

## 🌐 Deployment

### Vercel Deployment

This project is optimized for Vercel's serverless function deployment with a total size under 250MB.

**Optimization Details:**
- **Runtime dependencies** (`requirements.txt`): Only Flask, numpy, and Werkzeug (~80MB)
- **Lightweight model format**: JSON-based model without scikit-learn dependencies
- **Excluded files** (`.vercelignore`): Training scripts, notebooks, and development dependencies
- **Development dependencies** (`requirements-dev.txt`): Full ML stack for model training

**Steps to deploy:**

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel --prod
   ```

4. **Access your deployed app**
   The deployment URL will be displayed in the terminal

**Note:** The deployment uses the lightweight `house_price_model.json` file instead of the pickle file to avoid including scikit-learn in the serverless function.

## 💻 Usage

### Web Interface

1. Navigate to the web application
2. Enter the house details:
   - Overall Quality (1-10)
   - Living Area in square feet
   - Basement Area in square feet
   - Garage Capacity (number of cars)
   - Garage Area in square feet
   - Year Built
3. Click "Predict House Price"
4. View the predicted price and input summary

### API Usage

The application also provides a JSON API endpoint:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "OverallQual": 7,
    "GrLivArea": 2000,
    "TotalBsmtSF": 1200,
    "GarageCars": 2,
    "GarageArea": 500,
    "YearBuilt": 2010
  }'
```

## 🔧 Technologies Used

- **Backend**: Flask
- **Machine Learning**: scikit-learn (training only)
- **Data Processing**: pandas (training only), numpy (runtime)
- **Model Persistence**: 
  - Training: pickle (scikit-learn format)
  - Deployment: JSON (lightweight format)
- **Deployment**: Vercel (serverless functions)
- **Frontend**: HTML5, CSS3

## 📈 Model Performance

The model achieves strong performance metrics:
- High R² score indicating good fit
- Low RMSE for accurate predictions
- Consistent performance on test data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Project developed as part of the House Price Prediction System implementation.

## 📞 Support

For any questions or issues, please open an issue in the GitHub repository.

## 🔗 Links

- GitHub Repository: https://github.com/moladeji2302025-ctrl/House-Price-Prediction-System
- Live Demo: [To be updated after deployment]