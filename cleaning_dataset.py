import pandas as pd

df = pd.read_csv("daily_food_nutrition_dataset.csv")
agg = df.groupby("Food_Item", as_index=False).agg({
    "Calories (kcal)": "mean",
    "Protein (g)": "mean",
    "Carbohydrates (g)": "mean",
    "Fat (g)": "mean",
    "Fiber (g)": "mean",
    "Sugars (g)": "mean",
    "Category": "first"
})
agg.to_csv("cleaned_food_nutrition.csv", index=False)
