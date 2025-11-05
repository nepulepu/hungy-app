# ==========================================
# complementary_ai.py
# Classical AI Components for Hungy
# ==========================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1️⃣ Recipe Recommendation System (Cosine Similarity)
# ------------------------------------------------------------
class RecipeRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.recipe_texts = []
        self.sim_matrix = None

    def fit(self, recipe_texts):
        """
        Fit recommender with a list of recipe ingredient strings.
        e.g., ["chicken rice broccoli", "beef potato gravy", ...]
        """
        self.recipe_texts = recipe_texts
        X = self.vectorizer.fit_transform(recipe_texts)
        self.sim_matrix = cosine_similarity(X)

    def recommend(self, recipe_index, top_k=2):
        """
        Recommend top_k most similar recipes.
        Returns a list of tuples (recipe_name, similarity_score).
        """
        if self.sim_matrix is None:
            raise ValueError("Recommender not fitted yet. Call .fit() first.")

        scores = list(enumerate(self.sim_matrix[recipe_index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [
            (self.recipe_texts[idx], float(score))
            for idx, score in scores[1: top_k + 1]
        ]


# ------------------------------------------------------------
# 2️⃣ Rule-Based Health Scoring System
# ------------------------------------------------------------
def health_score(nutrition):
    """
    Compute a healthiness score (0–100) based on nutrition dictionary.
    Expected keys: calories, fat, sugar, fiber, protein
    """
    cal = nutrition.get("calories", 0)
    fat = nutrition.get("fat", 0)
    sugar = nutrition.get("sugar", 0)
    fiber = nutrition.get("fiber", 0)
    protein = nutrition.get("protein", 0)

    score = 100
    score -= (fat * 0.5 + sugar * 0.4)
    score += (fiber * 1.5 + protein * 1.2)
    score -= abs(cal - 400) * 0.05  # penalize very high/low calories

    return max(0, min(100, round(score)))


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example: Recipe Similarity
    recipes = [
        "chicken rice broccoli",
        "beef potato gravy",
        "grilled salmon spinach",
        "tofu stir fry vegetables",
        "apple banana smoothie",
    ]
    recommender = RecipeRecommender()
    recommender.fit(recipes)
    print(recommender.recommend(0))

    # Example: Health Score
    example = {"calories": 450, "fat": 10, "sugar": 5, "fiber": 4, "protein": 25}
    print("Health Score:", health_score(example))
