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
│   └── house_price_model.pkl      # Trained model (generated)
├── templates/
│   └── index.html                 # Web interface
├── static/                        # (Optional) CSS/JS files
├── app.py                         # Flask web application
├── requirements.txt               # Python dependencies
├── vercel.json                    # Vercel deployment configuration
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
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python model/model_development.py
   ```
   This will generate `model/house_price_model.pkl`

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
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
   - Saved using Python's pickle module
   - Includes model, scaler, and feature columns

## 🌐 Deployment

### Vercel Deployment

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
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Persistence**: pickle
- **Deployment**: Vercel
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