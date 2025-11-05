from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import numpy as np

# Step 1: Clean dataset
df = pd.read_csv("daily_food_nutrition_dataset.csv")
df = df.groupby("Food_Item", as_index=False).agg({
    "Calories (kcal)": "mean",
    "Protein (g)": "mean",
    "Carbohydrates (g)": "mean",
    "Fat (g)": "mean",
    "Fiber (g)": "mean",
    "Sugars (g)": "mean",
    "Category": "first"
}).dropna()

# Step 2: Get embeddings
model_emb = SentenceTransformer("all-MiniLM-L6-v2")
X = model_emb.encode(df["Food_Item"].tolist(), show_progress_bar=True)

# Step 3: Train per nutrient
targets = ["Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fat (g)", "Fiber (g)", "Sugars (g)"]
results = {}

for t in targets:
    y = df[t].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    results[t] = {"R2": r2, "MAE": mae}
    print(f"{t}: R²={r2:.3f}, MAE={mae:.2f}")
