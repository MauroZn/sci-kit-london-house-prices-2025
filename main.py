import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import time

start = time.time()

#WARNING!
#FILE "0_source_raw_data_kaggle_london_house_price_data.csv" IS NOT ON GIT ANYMORE since it was too large to store on the repo
#ONLY "1_specific_columns_london_house_prices.csv" is uploaded and is needed for this project.
#And this main.py is not really needed for the user anyway, but check the doc on Github for a better explanation
#-----------------------------------------------------------------------------------------------------------------------

##Used to get the important columns and create a new csv file (Not needed anymore now):
# df = pd.read_csv("0_source_raw_data_kaggle_london_house_price_data.csv")
# london_house_price_df = df.dropna()
# clean_df = london_house_price_df.drop_duplicates()
# useful_columns_df = clean_df[['fullAddress','postcode','floorAreaSqM','propertyType','rentEstimate_currentPrice','saleEstimate_currentPrice', 'history_date']].copy()
# useful_columns_df.to_csv('1_specific_columns_london_house_prices.csv', index=False)
# print(useful_columns_df.columns.to_list())

#This data is from 1995-01-02 to 2024-09-27
df = pd.read_csv("data/1_specific_columns_london_house_prices.csv")
df = df.dropna()
df = df.drop_duplicates()
clean_df = df.copy()
clean_df['history_date'] = pd.to_datetime(clean_df['history_date'])
clean_df['year'] = clean_df['history_date'].dt.year
clean_df['month'] = clean_df['history_date'].dt.month
clean_df['day'] = clean_df['history_date'].dt.day
#Temporary adjustment to get only 2023 to 2024 data so it will take less to do the training
clean_df = clean_df[clean_df['year'].between(2023, 2024)]
# clean_df.to_csv('2_final_cleaned_model_data_london_house_prices.csv', index=False)

print("Defining features and targets of the model...")
features = ['postcode', 'floorAreaSqM', 'propertyType', 'year', 'month']
X = clean_df[features]
y_rent = clean_df['rentEstimate_currentPrice']
y_sale = clean_df['saleEstimate_currentPrice']

print("Training the model only on past data...")
train_df = clean_df[clean_df['history_date'] < '2024-07-01']
test_df = clean_df[clean_df['history_date'] >= '2024-07-01']

X_train = train_df[features]
X_test = test_df[features]
y_rent_train = train_df['rentEstimate_currentPrice']
y_rent_test = test_df['rentEstimate_currentPrice']
y_sale_train = train_df['saleEstimate_currentPrice']
y_sale_test = test_df['saleEstimate_currentPrice']

categorical_cols = ['postcode', 'propertyType']
numerical_cols = ['floorAreaSqM', 'year', 'month']

#Scikit Learn
print("Starting Scikit Learn...")
print("Processing Data..")
#Apply different transformations to different columns:
#StandardScaler() normalizes numeric values (e.g., floorAreaSqM, year, month)
#OneHotEncoder() converts categorical features (e.g., postcode, propertyType) into binary columns (0/1) for machine learning
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

print("Creating the pipelines rent and sale..")
#RandomForestRegressor is the type of scikit model used for this particular case, it works well with tabular data.
rent_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=0))
])
sale_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=0))
])

print("Training...")
rent_pipeline.fit(X_train, y_rent_train)
sale_pipeline.fit(X_train, y_sale_train)

# Save trained models to disk
joblib.dump(rent_pipeline, 'models/rent_model.pkl')
joblib.dump(sale_pipeline, 'models/sale_model.pkl')

print("Models saved successfully.")
end = time.time()
end_to_minutes = round(end / 60, 0)
print("It took around" + str(end_to_minutes) + "minutes to train the models.")