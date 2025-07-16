<a id="readme-top"></a>
<div align="center">
<h1>Sci Kit London House Prices</h1>
<h2>Predicting Rent and Sale Prices for 2025</h2>
<i>This is my first bigger project made with Sci-kit and using RandomForestRegressor.</i> <br>
</div>
<hr>
<div style="text-align: justify;
    text-align-last: center;">

## Project Idea
This project is a machine learning pipeline that predicts London house rental and sale prices using historical housing data. It focuses on structured tabular data from 1995 to 2024 and leverages the power of Scikit-learn, particularly Random Forest Regressors, to estimate prices based on features like property size, type, location (postcode), and time. 

### Example Use Case:
A real estate agency wants to estimate the monthly rent and market value of a 75 m² flat in postcode W1A for each month of 2025. With this system, they can input the details once and get an interactive forecast graph showing how prices fluctuate throughout the year, enabling data-driven decision-making.
</div>

> [!IMPORTANT]
> **The models uploaded on Git are still not 100% completed since they only have the data from 2023 to 2024. This choice has been made to test the prediction without waiting too much time to train and create the models.**

<!-- TABLE OF CONTENTS -->
## Table of contents
1. [User Guide](#user-guide)
   - [Predict Single Property Single Month](#single-house-month)
   - [Predict Single Property All 2025](#single-house-year)
   - [Predict Average All Properties 2025](#average-all-houses)
2. [Code Analysis](#code-analysis)
   - [1. Load and Clean the Data](#code-1)
   - [2. Date Engineering](#code-2)
   - [3. Define Features and Targets](#code-3)
   - [4. Time-Based Train/Test Split](#code-4)
   - [5. Preprocessing Setup](#code-5)
   - [6. Pipeline Creation](#code-6)
   - [7. Model Training](#code-7)
   - [8. Save Trained Models](#code-8)
3. [Links and Manuals](#links)

> [!WARNING]
> **WORK IN PROGRESS! This documentation is still not finshed and needs to be reviewed.**

<hr>

<a id="user-guide"></a>
## User Guide

This application predicts monthly rent and sale prices for residential properties in London using a machine learning model trained on historical housing data from 2023–2024 (Only from 2023 for Test reasons).

There are three main functionalities:
- ```predict_house_price_single_month.py:``` Predict the price for a specific property in a specific month of 2025.
- ```predict_house_price_all_2025.py:``` Predict and Visualize the price for a specific property through all 2025.
- ```predict_2025_average_prices.py:``` Visualize how an average of all properties cost fluctuates throughout all 2025.

### Requirements
- Python 3.8+
- pandas, matplotlib, joblib, scikit-learn

Run:
```
pip install -r requirements.txt
```

### Files
| File Name                                               | Description                                                                     |
|---------------------------------------------------------|---------------------------------------------------------------------------------|
| ```main.py```                                                | Trains and saves the machine learning models (rent_model.pkl, sale_model.pkl).  |
| ```predict_house_price_single_month.py```                    | 	Predicts rent/sale price for one property in one month.                        |
| ```predict_house_price_all_2025.py```                         | Predicts and Visualize rent/sale price for one property throughout all 2025.    |
| ```predict_2025_average_prices.py```                          | Visualize rent/sale average price of all London properties throughout all 2025. |
| ```data/2_final_cleaned_model_data_london_house_prices.csv``` | Cleaned dataset used for training and prediction.                               |
| ```models/```                                               | Folder containing trained model files.                                          |
| ```requirements.txt```                                       | Text file used to install all the required Python packages                      |

### How to Use
### 0. Training the Models (OPTIONAL)
> [!NOTE]
> Models are already trained and saved in .pkl files in models/ folder so this part is **OPTIONAL**

If you want to retrain you can just run this code and wait for the models to be created (It could take around 1 hour):
```
python main.py
```
<a id="single-house-month"></a>
### 1. Predict Price for One Property for One Specific Month (Example: September 2025)
Run:
```
python predict_house_price_single_month.py
```

You’ll be prompted to enter:
- Postcode (or list all to choose)
- Floor area (in square meters)
- Property type (select from valid types)
- Month (1–12)

It returns:
- Predicted rent
- Predicted sale price

#### Example:
```
Predicted Rent: £2150.00
Predicted Sale Price: £475,000.00
```
<a id="single-house-year"></a>
### 2. Predict and Visualize Prices for One Property for All 2025:
Run:
```
python predict_house_price_all_2025.py
```
Like the single month prediction, you will be prompted to enter:
- Postcode (or list all to choose)
- Floor area (in square meters)
- Property type (select from valid types)

It returns:
- Rent Graph, showing the rent that the property will have throughout all 2025.
- Sale Graph, showing the sale price that the property will have throughout all 2025.

<a id="average-all-houses"></a>
### 3. Predict and Visualize Average Prices for All 2025:
Run:
```
python predict_2025_average_prices.py
```

It returns:
- Rent Graph, showing the generic average rent taking all the properties in London throughout all 2025.
- Sale Graph, showing the generic average sale price taking all the properties in London throughout all 2025.

<hr>

<a id="code-analysis"></a>
## Code Analysis
The project started with the use of a Kraggle Dataset containing London Rent and Sale Property Prices from 1995-2024, that has been downloaded and saved in a .csv file.

<a id="code-1"></a>
### 1. Load and Clean the Data
```
cleaned_final_df = pd.read_csv("data/1_specific_columns_london_house_prices.csv")
cleaned_final_df.dropna(inplace=True)
cleaned_final_df.drop_duplicates(inplace=True)
```
Remove rows with missing values ```(dropna())``` and eliminate duplicate rows ```(drop_duplicates())```

<a id="code-2"></a>
### 2. Date Engineering
```
cleaned_final_df['history_date'] = pd.to_datetime(cleaned_final_df['history_date'])
cleaned_final_df['year'] = cleaned_final_df['history_date'].dt.year
cleaned_final_df['month'] = cleaned_final_df['history_date'].dt.month
cleaned_final_df['day'] = cleaned_final_df['history_date'].dt.day
```
Convert date strings into datetime format and extract year, month, and day for use as predictive features.

### Filter recent years (Only for Test Purpose)
```
# Temporary adjustment to get only 2023 to 2024 data so it will take less to do the training
cleaned_final_df = cleaned_final_df[cleaned_final_df['year'].between(2023, 2024)]
```

<a id="code-3"></a>
### 3. Define Features and Targets
```
features = ['postcode', 'floorAreaSqM', 'propertyType', 'year', 'month']
X = cleaned_final_df[features]
y_rent = cleaned_final_df['rentEstimate_currentPrice']
y_sale = cleaned_final_df['saleEstimate_currentPrice']
```
- X: Feature matrix (independent variables).
- y_rent: Target for rental price prediction.
- y_sale: Target for sale price prediction.
- 
<a id="code-4"></a>
### 4. Time-Based Train/Test Split
```
train_df = cleaned_final_df[cleaned_final_df['history_date'] < '2024-07-01']
test_df = cleaned_final_df[cleaned_final_df['history_date'] >= '2024-07-01']
```

Predict future prices based on past data and ensures that the model is not trained on future data (avoids data leakage)

### Separate training and testing data for each prediction target
```
X_train = train_df[features]
X_test = test_df[features]
y_rent_train = train_df['rentEstimate_currentPrice']
y_rent_test = test_df['rentEstimate_currentPrice']
y_sale_train = train_df['saleEstimate_currentPrice']
y_sale_test = test_df['saleEstimate_currentPrice']
```

<a id="code-5"></a>
### 5. Preprocessing Setup
```
categorical_cols = ['postcode', 'propertyType']
numerical_cols = ['floorAreaSqM', 'year', 'month']
```
- Categorical: postcode, propertyType
- Numerical: floorAreaSqM, year, month

### Pre Processor
```
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])
``` 

### ColumnTransformer:
Used to apply appropriate transformations:
- **StandardScaler()** normalizes numerical data to mean 0 and variance 1.
- **OneHotEncoder()** transforms categorical data into binary columns.

<a id="code-6"></a>
### 6. Pipeline Creation
``` 
rent_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=0))
])
sale_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=0))
])
``` 

- Ensures all preprocessing happens during both training and inference.
- Prevents data leakage.
- Makes model easier to deploy and save.
### RandomForestRegressor: 
An ensemble machine learning model that builds multiple decision trees during training and averages their predictions to improve accuracy and reduce overfitting.<br>
It works well with structured tabular data, can handle both numerical and categorical features (after encoding), and is robust to noise and outliers. 

<a id="code-7"></a>
### 7. Model Training
``` 
rent_pipeline.fit(X_train, y_rent_train)
sale_pipeline.fit(X_train, y_sale_train)
```
Each pipeline trains on preprocessed data to predict rent and sale prices.

<a id="code-8"></a>
### 8. Save Trained Models
```
joblib.dump(rent_pipeline, 'models/rent_model.pkl')
joblib.dump(sale_pipeline, 'models/sale_model.pkl')
```

### Joblib 
It is a Python library optimized for efficiently serializing and deserializing large objects like machine learning models.
In this case it saves the entire trained pipeline (preprocessing + model) to disk, allowing you to reload and use the model later without retraining

> [!NOTE]
> I used Git LFS to store the two .pkl files inside my Git Repo.
> I decided to go with this solution since the files are too heavy and Git doesn't accept them without LFS.

<hr>

## Pipeline Scheme:<br>
 Raw input X (with categorical and numeric features)<br>
         ↓<br>
 [Pipeline]<br>
   ├── Step 1: Preprocessor (ColumnTransformer)<br>
   │     ├── StandardScaler on floorAreaSqM, year, month<br>
   │     └── OneHotEncoder on postcode, propertyType<br>
   ↓<br>
   Transformed Feature Matrix (all numeric)<br>
         ↓<br>
   └── Step 2: RandomForestRegressor<br>
         ↓<br>
   Predicted rent or sale prices<br>

<hr>

<a id="links"></a>
## Links and Manuals
### Kraggle Dataset
- [Kraggle Original Dataset](https://www.kaggle.com/datasets/jakewright/house-price-data)
### Sci-Kit Full Documentation
- [Sci-Kit Full Documentation](https://scikit-learn.org/stable/)
#### Used in this project:
- [Sci-Kit ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Sci-Kit StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Sci-Kit OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Sci-Kit RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<hr>
