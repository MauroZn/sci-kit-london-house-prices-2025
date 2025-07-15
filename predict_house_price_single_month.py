import joblib
import pandas as pd
from collections import defaultdict

rent_pipeline = joblib.load('models/rent_model.pkl')
sale_pipeline = joblib.load('models/sale_model.pkl')

clean_df = pd.read_csv("data/2_final_cleaned_model_data_london_house_prices.csv")

#Default Values
postcode = 'E1 6AN'
floorAreaSqm = 65
propertyType = 'Flat'
month = 1

property_types = (
    'Purpose Built Flat', 'Semi-Detached Bungalow', 'Semi-Detached House',
    'Terrace Property', 'Mid Terrace House', 'Converted Flat', 'Detached House',
    'Flat/Maisonette', 'End Terrace House', 'Terraced', 'Bungalow Property',
    'Detached Bungalow', 'Semi-Detached Property', 'End Terrace Property',
    'Terraced Bungalow', 'Detached Property', 'End Terrace Bungalow',
    'Mid Terrace Bungalow', 'Mid Terrace Property'
)

print("PREDICT HOUSE PRICES IN LONDON:\n"
      "Description: This app will predict the rent and sale cost of a house in London, based on: Postcode, Floor Area, Property Type and Month of the year 2025.\n"
      "----------------------------------------------------------------------------------------------------------------------------------------------")

def choose_postcode():
    global postcode
    print("Postcode Section:")
    input_show_postcodes = input("Do you wanna see all postcodes to choose one? If not just leave the 'Enter a postcode: ' field empty after this(Choose 'yes' or 'no'): ")
    if input_show_postcodes == 'yes':
        grouped = defaultdict(list)
        for p in clean_df['postcode'].unique():
            prefix = p.split()[0][:2]
            grouped[prefix].append(p)

        print("Available postcode groups:")
        for prefix in sorted(grouped):
            print(f"{prefix}: {len(grouped[prefix])} postcodes")

        user_choice = input("\nEnter a postcode prefix to view all matching postcodes (e.g., E1): ").upper()
        if user_choice in grouped:
            print(f"\nPostcodes for {user_choice}:")
            for postcode in grouped[user_choice]:
                print(postcode)
        else:
            print(f"\n'{user_choice}' is not a valid prefix.")

        input_postcode = input('Enter a postcode: ').upper()
        if input_postcode != '':
            postcode = input_postcode

def choose_floor_area():
    global floorAreaSqm
    print("Floor Area Section:")
    input_floor_area_sqm = input('Enter the floor area sqm: ')
    if input_floor_area_sqm != '':
        floorAreaSqm = float(input_floor_area_sqm)

def choose_property_type():
    global propertyType
    print("Property Type Section:")
    print("These are the property types available:")
    for p in property_types:
        print("-" + p)
    input_property_type = input('Enter the property type (Default: Flat): ')
    if input_property_type != '':
        propertyType = input_property_type


def choose_month():
    global month
    print("Month of 2025 Section:")
    input_month = input('Enter the month (Default: 1): ')
    if input_month != '':
        month = int(input_month)

choose_postcode()
choose_property_type()
choose_floor_area()
choose_month()

future_data = pd.DataFrame([{
    'postcode': postcode,
    'floorAreaSqM': floorAreaSqm,
    'propertyType': propertyType,
    'year': 2025,
    'month': month
}])

pred_rent = rent_pipeline.predict(future_data)[0]
pred_sale = sale_pipeline.predict(future_data)[0]

print(f"Predicted Rent for a {future_data['propertyType'].iloc[0]} of {future_data['floorAreaSqM'].iloc[0]} SqM in the {future_data['postcode'].iloc[0]} postcode on the {future_data['month'].iloc[0]} month of {future_data['year'].iloc[0]}: £{pred_rent:.2f}")
print(f"Predicted Sale Price for a {future_data['propertyType'].iloc[0]} of {future_data['floorAreaSqM'].iloc[0]} SqM in the {future_data['postcode'].iloc[0]} postcode on the {future_data['month'].iloc[0]} month of {future_data['year'].iloc[0]}:  £{pred_sale:.2f}")
