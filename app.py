import gradio as gr
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load upgraded XGBoost model
model = joblib.load("car_price_model_xgb.pkl")

# Load dataset
df = pd.read_csv("car_data_with_features.csv")
features = ['Power_BHP', 'Mileage_kmpl', 'Brand_Goodwill']
target = 'Selling_Price'

X = df[features]
y = df[target]

# Predictions for performance metrics
y_pred = model.predict(X)

# Evaluation metrics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Top 10 predictions table
comparison = pd.DataFrame({
    'Car Name': df['Car_Name_old'],
    'Actual Price': y,
    'Predicted Price': y_pred
}).sort_values(by='Actual Price', ascending=False).head(10)

# Charts
import matplotlib.pyplot as plt
import numpy as np

# Correct Chart Functions
def plot_actual_vs_pred():
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, color='#1f77b4', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Price (Lakhs)")
    ax.set_ylabel("Predicted Price (Lakhs)")
    ax.set_title("Actual vs Predicted Prices")
    fig.tight_layout()
    return fig  # <-- Return the figure object directly

def plot_error_distribution():
    errors = y - y_pred
    fig, ax = plt.subplots()
    ax.hist(errors, bins=30, color='#ff7f0e', edgecolor='black')
    ax.set_xlabel("Prediction Error (Lakhs)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")
    fig.tight_layout()
    return fig  # <-- Return figure directly
# Prediction function
def predict_price(power, mileage, goodwill):
    input_df = pd.DataFrame([{
        "Power_BHP": power,
        "Mileage_kmpl": mileage,
        "Brand_Goodwill": goodwill
    }])
    prediction = model.predict(input_df)[0]
    confidence = max(60, round(100 - abs(prediction - goodwill)*1.5, 2))
    return f"🚗 **Predicted Price:** {prediction:.2f} Lakhs\n📊 **Confidence Score:** {confidence}%"

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🚗 AI Car Price Predictor (XGBoost)")

    with gr.Group():
        gr.Markdown("## 🔮 Predict Car Price")
        with gr.Row():
            power = gr.Slider(40, 300, value=100, label="⚡ Power (BHP)")
            mileage = gr.Slider(5, 40, value=20, label="⛽ Mileage (KMPL)")
        goodwill = gr.Slider(1, 50, value=10, label="🏆 Brand Goodwill")
        predict_btn = gr.Button("🚀 Predict")
        output = gr.Markdown()
        predict_btn.click(fn=predict_price, inputs=[power, mileage, goodwill], outputs=output)

    with gr.Group():
        gr.Markdown("## 📊 Model Performance")
        gr.Markdown(f"**MAE:** {mae:.2f} Lakhs  \n**RMSE:** {rmse:.2f} Lakhs  \n**R²:** {r2:.2f}")
        gr.Markdown("### 🔝 Top 10 Predictions")
        gr.Dataframe(value=comparison, interactive=False)

    with gr.Group():
        gr.Markdown("## 📈 Charts")
        with gr.Row():
            chart1 = gr.Plot(value=plot_actual_vs_pred)
            chart2 = gr.Plot(value=plot_error_distribution)

if __name__ == "__main__":
    app.launch()