import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import defaultdict

rent_pipeline = joblib.load('models/rent_model.pkl')
sale_pipeline = joblib.load('models/sale_model.pkl')

clean_df = pd.read_csv("data/2_final_cleaned_model_data_london_house_prices.csv")

#Default Values
postcode = 'E1 6AN'
floorAreaSqM = 65
propertyType = 'Flat'

property_types = (
    'Purpose Built Flat', 'Semi-Detached Bungalow', 'Semi-Detached House',
    'Terrace Property', 'Mid Terrace House', 'Converted Flat', 'Detached House',
    'Flat/Maisonette', 'End Terrace House', 'Terraced', 'Bungalow Property',
    'Detached Bungalow', 'Semi-Detached Property', 'End Terrace Property',
    'Terraced Bungalow', 'Detached Property', 'End Terrace Bungalow',
    'Mid Terrace Bungalow', 'Mid Terrace Property'
)

print("\nPREDICT 2025 MONTHLY PRICE TRENDS FOR A SPECIFIC LONDON HOUSE")
print("----------------------------------------------------------------")

def choose_postcode():
    global postcode
    show_postcodes = input("Do you want to view available postcodes? (yes/no): ")
    if show_postcodes.lower() == 'yes':
        grouped = defaultdict(list)
        for p in clean_df['postcode'].unique():
            prefix = p.split()[0][:2]
            grouped[prefix].append(p)

        for prefix in sorted(grouped):
            print(f"{prefix}: {len(grouped[prefix])} postcodes")

        prefix_input = input("\nEnter a postcode prefix to see full list (e.g., E1): ").upper()
        if prefix_input in grouped:
            print(f"Postcodes under {prefix_input}:")
            for p in grouped[prefix_input]:
                print(p)

        user_postcode = input("Enter the postcode: ").upper()
        if user_postcode:
            postcode = user_postcode

def choose_floor_area():
    global floorAreaSqM
    user_area = input("Enter floor area in sqm (Default 65): ")
    if user_area:
        floorAreaSqM = float(user_area)

def choose_property_type():
    global propertyType
    print("Available property types:")
    for p in property_types:
        print("- " + p)
    user_type = input("Enter property type (Default: Flat): ")
    if user_type:
        propertyType = user_type


choose_postcode()
choose_floor_area()
choose_property_type()

# Generate data for every month of 2025
monthly_data = pd.DataFrame([{
    'postcode': postcode,
    'floorAreaSqM': floorAreaSqM,
    'propertyType': propertyType,
    'year': 2025,
    'month': m
} for m in range(1, 13)])


monthly_data['predicted_rent'] = rent_pipeline.predict(monthly_data)
monthly_data['predicted_sale'] = sale_pipeline.predict(monthly_data)

# Rent
plt.figure(figsize=(10, 5))
plt.plot(monthly_data['month'], monthly_data['predicted_rent'], marker='o', color='green')
plt.title(f"Predicted Rent for {propertyType} in {postcode} (2025)")
plt.xlabel("Month")
plt.ylabel("Rent Price (£)")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.show()

# Sale
plt.figure(figsize=(10, 5))
plt.plot(monthly_data['month'], monthly_data['predicted_sale'], marker='o', color='blue')
plt.title(f"Predicted Sale Price for {propertyType} in {postcode} (2025)")
plt.xlabel("Month")
plt.ylabel("Sale Price (£)")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.show()
