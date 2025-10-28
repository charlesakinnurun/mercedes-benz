# Mercedes-Benz
![Mercedes Benz](/image.jpg)

## Overview
The goal is to predict the time it takes a mercedes-benz given the neccessary parameters. The workflow includes Data loading, Data preprocessing, Feature engineering, Data encoding, Data splitting and lots more.

## Procedures
- Data Loading
- Data Preprocessing
    - Handle missing values
    - Handle duplicated rows
- Feature Engineering
- Data Encoding
    - One-Hot Encoding
    - Label Encoding
    - Column Transformer
- Data Splitting
    - 80% for training and 20% for testing
- Pre-Training Visualization
    - Distribution of Target Variable using histplot
- Model Training and Comparison
    - Linear Regression
    - Lasso Regression
    - Ridge Regression
    - Elastic Net Regression
    - XGBoost Regression
    - Random Forest Regression
- Hyperparameter Tuning
    - GridSearchCV
- Model 
    - R2 Score
    - Root Mean Squared Score
- Post-Training Visualization
    - Actual vs Predicted Values using Line plot
- New Prediction Input

### Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/mercedes-benz.git
cd mercedes-benz
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Tools and Dependencies
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - xgboost
- Environment
    - Jupyter Notebook
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```

## Project Structure
```
mercedes-benz/
│
├── model.ipynb  
|── model.py    
|── submission_predictions.csv  
├── requirements.txt 
├── LICENSE
├── image.jpg    
├── test.csv
|──   train.csv
└── README.md          
```
## Contributing
Contributions are welcome! If you’d like to suggest improvements — e.g., new modelling algorithms, additional feature engineering, or better documentation — please open an Issue or submit a Pull Request.
Please ensure your additions are accompanied by clear documentation and, where relevant, updated evaluation results.

## License
This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.