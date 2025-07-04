<a id="readme-top"></a>
<div align="center">
<h1>Sci Kit Random Forest Regressor</h1>
<h2>Predict London House Prices 2025</h2>
</div>
<hr>

<!-- TABLE OF CONTENTS -->
## Table of contents
1. [User Guide](#user-guide)
2. [Code Description and Idea](#description)
3. [Links and Manuals](#links)

> [!WARNING]
> **WORK IN PROGRESS! This documentation is still not finshed, I just started doing it.**

<hr>

<a id="user-guide"></a>
## User Guide
Only use files predict.py or predict_all_2025_prices.py.

<hr>

<a id="description"></a>
## Description
- Used Kraggle Dataset and Sci-kit to predict London House Prices of 2025.
- First project made with Sci-kit (Except two lessons of the udemy course)
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
