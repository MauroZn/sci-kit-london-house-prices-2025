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
#-----------------------------------------------------------------------------------------------------------------------

## Used to extract the important columns and save them to a smaller CSV file:
## NOT NEEDED ANYMORE NOW since the file 1_specific_columns_london_house_prices.csv is already in the data folder of this project and there is no need to create it again
# df_original = pd.read_csv("0_source_raw_data_kaggle_london_house_price_data.csv")
# df_original.dropna(inplace=True)
# df_original.drop_duplicates(inplace=True)
# useful_columns = ['fullAddress', 'postcode', 'floorAreaSqM', 'propertyType',
#                   'rentEstimate_currentPrice', 'saleEstimate_currentPrice', 'history_date']
# df_original[useful_columns].to_csv('1_specific_columns_london_house_prices.csv', index=False)
# print(useful_columns)

# This data is from 1995-01-02 to 2024-09-27
cleaned_final_df = pd.read_csv("data/1_specific_columns_london_house_prices.csv")
cleaned_final_df.dropna(inplace=True)
cleaned_final_df.drop_duplicates(inplace=True)
cleaned_final_df['history_date'] = pd.to_datetime(cleaned_final_df['history_date'])
cleaned_final_df['year'] = cleaned_final_df['history_date'].dt.year
cleaned_final_df['month'] = cleaned_final_df['history_date'].dt.month
cleaned_final_df['day'] = cleaned_final_df['history_date'].dt.day

# Temporary adjustment to get only 2023 to 2024 data so it will take less to do the training
cleaned_final_df = cleaned_final_df[cleaned_final_df['year'].between(2023, 2024)]
# cleaned_final_df.to_csv('2_final_cleaned_model_data_london_house_prices.csv', index=False)

print("Defining features and targets of the model...")
features = ['postcode', 'floorAreaSqM', 'propertyType', 'year', 'month']
X = cleaned_final_df[features]
y_rent = cleaned_final_df['rentEstimate_currentPrice']
y_sale = cleaned_final_df['saleEstimate_currentPrice']

print("Training the model only on past data...")
train_df = cleaned_final_df[cleaned_final_df['history_date'] < '2024-07-01']

# I still didn't do the Evaluation of the model after the training, so these variables will not be used for now but stored here for later use.
# test_df = cleaned_final_df[cleaned_final_df['history_date'] >= '2022-07-01']
# y_sale_test = test_df['saleEstimate_currentPrice']
# y_rent_test = test_df['rentEstimate_currentPrice']
# X_test = test_df[features]

X_train = train_df[features]
y_rent_train = train_df['rentEstimate_currentPrice']
y_sale_train = train_df['saleEstimate_currentPrice']


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