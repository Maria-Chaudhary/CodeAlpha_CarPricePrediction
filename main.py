# -------------------------------
# 1️⃣ Load datasets and merge features (your existing fuzzy logic)
# -------------------------------
import pandas as pd
from fuzzywuzzy import process

# Load original dataset
old_df = pd.read_csv("car data.csv")
new_df = pd.read_csv("final_cars_dataset.csv")  # Car_Name, Brand, Power_BHP, Mileage_kmpl, Price_Lakhs

# Clean Car_Name
old_df['Car_Name'] = old_df['Car_Name'].str.lower().str.strip()
new_df['Car_Name'] = new_df['Car_Name'].str.lower().str.strip()

# Fuzzy matching
def get_best_match(car_name, choices, threshold=70):
    match, score = process.extractOne(car_name, choices)
    if score >= threshold:
        return match
    return None

new_car_names = new_df['Car_Name'].tolist()
old_df['Matched_Car'] = old_df['Car_Name'].apply(lambda x: get_best_match(x, new_car_names))

# Merge fuzzy-matched features
df = pd.merge(
    old_df,
    new_df[['Car_Name','Power_BHP','Mileage_kmpl','Price_Lakhs']],
    left_on='Matched_Car',
    right_on='Car_Name',
    how='left',
    suffixes=('_old','_new')
)

# Extract Brand and compute Brand Goodwill
df['Brand'] = df['Matched_Car'].apply(lambda x: x.split()[0].lower() if pd.notnull(x) else 'unknown')
new_df['Brand'] = new_df['Brand'].str.lower().str.strip()
brand_avg_price = new_df.groupby('Brand')['Price_Lakhs'].mean().to_dict()
df['Brand_Goodwill'] = df['Brand'].map(brand_avg_price)
df['Brand_Goodwill'] = df['Brand_Goodwill'].fillna(new_df['Price_Lakhs'].mean())

# Fill missing numeric values
df['Power_BHP'] = df['Power_BHP'].fillna(df['Power_BHP'].mean())
df['Mileage_kmpl'] = df['Mileage_kmpl'].fillna(df['Mileage_kmpl'].mean())

# Save prepared dataset (optional)
df.to_csv("car_data_with_features.csv", index=False)

# -------------------------------
# 2️⃣ Train model using ONLY 3 features (XGBoost)
# -------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Use XGBoost Regressor
from xgboost import XGBRegressor

# Define target and features
target = 'Selling_Price'
features = ['Power_BHP', 'Mileage_kmpl', 'Brand_Goodwill']

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with scaler + XGBoost
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred)/y_test)) * 100

print("\n----- Model Evaluation -----")
print(f"Mean Absolute Error: {mae:.2f} Lakhs")
print(f"Root Mean Squared Error: {rmse:.2f} Lakhs")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Sample predictions
comparison = pd.DataFrame({
    'Car_Name': df.loc[y_test.index, 'Car_Name_old'],
    'Actual_Price': y_test,
    'Predicted_Price': y_pred
}).sort_values(by='Actual_Price', ascending=False)

print("\n----- Sample Predictions -----")
print(comparison.head(10))

# -------------------------------
# 3️⃣ Interactive prediction function using only 3 features
# -------------------------------
def predict_new_car_loop(model):
    """
    Predict car selling price interactively using Power, Mileage, Brand Goodwill.
    """
    print("\n===== Welcome to Car Selling Price Predictor =====")
    print("You only need to enter:")
    print("1️⃣ Power (BHP)")
    print("2️⃣ Mileage (KMPL)")
    print("3️⃣ Brand Goodwill (Lakhs)\n")

    while True:
        # Power
        while True:
            try:
                power_bhp = float(input("Power (BHP, e.g., 74.5): "))
                if power_bhp > 0:
                    break
                else:
                    print("⚠️  Power must be positive.")
            except ValueError:
                print("⚠️  Invalid input! Please enter a number.")

        # Mileage
        while True:
            try:
                mileage_kmpl = float(input("Mileage (KMPL, e.g., 24.0): "))
                if mileage_kmpl > 0:
                    break
                else:
                    print("⚠️  Mileage must be positive.")
            except ValueError:
                print("⚠️  Invalid input! Please enter a number.")

        # Brand Goodwill
        while True:
            try:
                brand_goodwill = float(input("Brand Goodwill (Lakhs, e.g., 8.5): "))
                if brand_goodwill > 0:
                    break
                else:
                    print("⚠️  Brand Goodwill must be positive.")
            except ValueError:
                print("⚠️  Invalid input! Please enter a number.")

        # Prepare DataFrame
        new_car = pd.DataFrame([{
            'Power_BHP': power_bhp,
            'Mileage_kmpl': mileage_kmpl,
            'Brand_Goodwill': brand_goodwill
        }])

        # Predict
        predicted_price = model.predict(new_car)[0]
        print(f"\n✅ Predicted Selling Price: {predicted_price:.2f} Lakhs")
        print("========================================\n")

        # Ask again
        while True:
            again = input("Do you want to predict another car? (Y/N): ").strip().upper()
            if again in ['Y', 'N']:
                break
            else:
                print("⚠️  Please enter 'Y' or 'N'.")

        if again == 'N':
            print("\n🎉 Thank you for using the Car Price Predictor!")
            break

df = pd.read_csv("car_data_with_features.csv")
print("Number of rows:", df.shape[0])
# Call the interactive loop
predict_new_car_loop(model)

# -------------------------------
# 4️⃣ Save the trained model
# -------------------------------
import joblib
joblib.dump(model, "car_price_model_xgb.pkl")
print("✅ XGBoost Model saved successfully!")