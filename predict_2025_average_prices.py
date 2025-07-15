import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load models
rent_pipeline = joblib.load('models/rent_model.pkl')
sale_pipeline = joblib.load('models/sale_model.pkl')

# Load full cleaned dataset
df = pd.read_csv("data/2_final_cleaned_model_data_london_house_prices.csv")
df = df.dropna().drop_duplicates()
df['history_date'] = pd.to_datetime(df['history_date'])

# Generate all rows for 2025 (replicating each house 12 times for each month)
features = ['postcode', 'floorAreaSqM', 'propertyType', 'year', 'month']
houses_2025 = []

for month in range(1, 13):
    month_df = df.copy()
    month_df['year'] = 2025
    month_df['month'] = month
    houses_2025.append(month_df[features])

all_houses_2025 = pd.concat(houses_2025, ignore_index=True)

# Predict rent and sale
all_houses_2025['predicted_rent'] = rent_pipeline.predict(all_houses_2025)
all_houses_2025['predicted_sale'] = sale_pipeline.predict(all_houses_2025)

# Aggregate: Average rent and sale per month
monthly_avg = all_houses_2025.groupby('month')[['predicted_rent', 'predicted_sale']].mean().reset_index()

# Plot Average Rent Prices
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg['month'], monthly_avg['predicted_rent'], color='green', marker='o')
plt.title('Average Predicted Rent Prices in London (2025)')
plt.xlabel('Month')
plt.ylabel('Rent Price (£)')
plt.grid(True)
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

# Plot Average Sale Prices
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg['month'], monthly_avg['predicted_sale'], color='blue', marker='o')
plt.title('Average Predicted Sale Prices in London (2025)')
plt.xlabel('Month')
plt.ylabel('Sale Price (£)')
plt.grid(True)
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()
