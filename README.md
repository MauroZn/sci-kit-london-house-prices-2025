<a id="readme-top"></a>
<div align="center">
<h1>Sci Kit London House Prices</h1>
<h2>Predicting Rent and Sale Prices for 2025</h2>
</div>
<hr>
<div style="text-align: justify;
    text-align-last: center;">

## Project Idea
This project is a machine learning pipeline that predicts London house rental and sale prices using historical housing data. It focuses on structured tabular data from 1995 to 2024 and leverages the power of Scikit-learn, particularly Random Forest Regressors, to estimate prices based on features like property size, type, location (postcode), and time. 


> [!WARNING]
> **The models uploaded on Git are still small with data from 2023 to 2024. This choice has been made to test the prediction without waiting too much time to create the final models.**


### Example Use Case:
A real estate agency wants to estimate the monthly rent and market value of a 75 m² flat in postcode W1A for each month of 2025. With this system, they can input the details once and get an interactive forecast graph showing how prices fluctuate throughout the year, enabling data-driven decision-making.
</div>

<!-- TABLE OF CONTENTS -->
## Table of contents
1. [User Guide](#user-guide)
   - [Predict Single Property Single Month](#single-house-month)
   - [Predict Single Property All 2025](#single-house-year)
   - [Predict Average All Properties 2025](#average-all-houses)
2. [Code Description](#description)
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
### 1. Training the Models (optional)
> [!NOTE]
> Models are already trained and saved in .pkl files in models/ folder so this part is **OPTIONAL**

If you want to retrain you can just run this code and wait for the models to be created (It could take around 1 hour):
```
python main.py
```
<a id="single-house-month"></a>
### 2. Predict Price for One Property for One Specific Month (Example: September 2025)
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
### 3. Predict and Visualize Prices for One Property for All 2025:
Run:
```
python predict_house_price_all_2025.py
```
Like the single month prediction, you will be prompted to enter:
- Postcode (or list all to choose)
- Floor area (in square meters)
- Property type (select from valid types)
- Month (1–12)

It returns:
- Rent Graph, showing the rent that the property will have throughout all 2025.
- Sale Graph, showing the sale price that the property will have throughout all 2025.

<a id="average-all-houses"></a>
### 4. Predict and Visualize Average Prices for All 2025:
Run:
```
python predict_2025_average_prices.py
```

It returns:
- Rent Graph, showing the generic average rent taking all the properties in London throughout all 2025.
- Sale Graph, showing the generic average sale price taking all the properties in London throughout all 2025.


<hr>

> [!CAUTION]
> **WORK IN PROGRESS! From here the documentation is still not finished, there are just some notes for myself.**


<a id="description"></a>
## Description
- Used Kraggle Dataset and Sci-kit to predict London House Prices of 2025.
- First project made with Sci-kit
- Used Git LFS to store the .pkl files since they are too big to push them on Git directly.

Code:
- StandardScaler() normalizes numeric values (e.g., floorAreaSqM, year, month)
- OneHotEncoder() converts categorical features (e.g., postcode, propertyType) into binary columns (0/1) for machine learning
- PIPELINE:<br>
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

- RandomForestRegressor is the type of scikit model used for this particular case, it works well with tabular data.

<hr>

<a id="links"></a>
## Links and Manuals
- [Sci-Kit Documentation](https://scikit-learn.org/stable/)
- [Kraggle Original Dataset](https://www.kaggle.com/datasets/jakewright/house-price-data)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<hr>
