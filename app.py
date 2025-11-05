import streamlit as st
import google.generativeai as genai
from PIL import Image
from datetime import datetime
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
import re


# ------------------------------------------------------------
# COMPLEMENTARY AI MODULE
# ------------------------------------------------------------
class RecipeRecommender:
    """Recommends similar recipes using TF-IDF + cosine similarity"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.recipe_texts = []
        self.sim_matrix = None

    def fit(self, recipe_texts):
        self.recipe_texts = recipe_texts
        X = self.vectorizer.fit_transform(recipe_texts)
        self.sim_matrix = cosine_similarity(X)

    def recommend(self, recipe_index, top_k=3):
        if self.sim_matrix is None:
            return []
        scores = list(enumerate(self.sim_matrix[recipe_index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[1:top_k + 1]  # return just indices and scores


def health_score(nutrition: Dict[str, float]) -> int:
    cal = nutrition.get("calories", 0)
    fat = nutrition.get("fat", 0)
    sugar = nutrition.get("sugar", 0)
    fiber = nutrition.get("fiber", 0)
    protein = nutrition.get("protein", 0)
    carbs = nutrition.get("carbohydrates", 0)

    score = 50  # start neutral

    # 1️⃣ Calorie balance (ideal: ~400 per meal)
    cal_diff = abs(cal - 400)
    if cal_diff <= 100:
        score += 10  # within ideal range
    elif cal_diff <= 200:
        score += 5
    else:
        score -= (cal_diff - 200) * 0.05  # penalty grows with deviation

    # 2️⃣ Fat & sugar penalties
    score -= fat * 0.8
    score -= sugar * 1.2

    # 3️⃣ Carbohydrate balance (mild penalty for excess)
    if carbs > 60:
        score -= (carbs - 60) * 0.4

    # 4️⃣ Protein & fiber rewards (density-based, capped)
    if cal > 0:
        protein_density = protein / (cal / 100)
        fiber_density = fiber / (cal / 100)

        # cap bonuses to avoid overrewarding dense foods
        protein_bonus = min(protein_density, 4) / 4 * 20  # up to +20 points
        fiber_bonus   = min(fiber_density, 2) / 2 * 10    # up to +10 points

        score += protein_bonus + fiber_bonus

    # 5️⃣ Final clamping and rounding
    score = max(0, min(100, round(score)))
    return score


# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title=" Hungy", page_icon="🍳", layout="wide")

@st.cache_resource
def init_gemini():
    """Initialize Gemini AI"""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "your-gemini-api-key-here")
        if api_key == "your-gemini-api-key-here":
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

@st.cache_data(ttl=300)
def check_api_status():
    """Cached status check (5min)"""
    model = init_gemini()
    if not model:
        return False
    try:
        test = model.generate_content("hi")
        return bool(test.text)
    except Exception:
        return False


# ------------------------------------------------------------
# GEMINI FUNCTIONS
# ------------------------------------------------------------
def identify_ingredients_from_image(image):
    """Use Gemini Vision to identify ingredients"""
    model = init_gemini()
    if model is None:
        st.error("❌ Gemini API not configured.")
        return []

    try:
        prompt = "List visible food ingredients in this image as a comma-separated list only."
        response = model.generate_content([prompt, image])
        if response.text:
            return [x.strip() for x in response.text.split(",") if x.strip()]
    except Exception as e:
        st.error(str(e))
    return []

def generate_recipe_and_nutrition(ingredients: List[str], restrictions: str = "", cuisine: str = "", difficulty: str = "medium"):
    """Ask Gemini to generate recipe *and* nutrition info"""
    model = init_gemini()
    if model is None:
        return "❌ Gemini not configured.", {}

    ingredients_text = ", ".join(ingredients)
    prompt = f"""
    You are a professional chef.
    Given the following ingredients: {ingredients_text}

    Create a {difficulty}-level recipe in {cuisine or "any"} style.
    Respect these dietary restrictions: {restrictions or "None"}.

    Provide the response in this format:

    **Recipe Name**
    Prep: X min | Cook: X min | Serves: X

    **Ingredients:**
    - list each with quantity

    **Steps:**
    1. ...
    2. ...

    **Estimated Nutrition per Serving:**
    Calories: ___ kcal
    Protein: ___ g
    Carbohydrates: ___ g
    Fat: ___ g
    Fiber: ___ g
    Sugars: ___ g

    **Tips:** short cooking advice
    """

    try:
        response = model.generate_content(prompt)
        text = response.text or "No response."

        # ------------------------------------------------------------
        # 🧠 Extract recipe name (usually the first bolded line)
        # ------------------------------------------------------------
        name = None
        match = re.search(r"\*\*(.+?)\*\*", text)  # looks for **Recipe Name**
        if match:
            name = match.group(1).strip()
        else:
            # fallback: take the first non-empty line as name
            for line in text.splitlines():
                line = line.strip()
                if line:
                    name = line
                    break
        if not name:
            name = "Unnamed Recipe"

        # ------------------------------------------------------------
        # 🧮 Extract nutrition estimates (basic parsing)
        # ------------------------------------------------------------
        nutrition = {}
        for key in ["Calories", "Protein", "Carbohydrates", "Fat", "Fiber", "Sugars"]:
            lower = key.lower()
            for line in text.splitlines():
                if lower in line.lower():
                    try:
                        value = float("".join([c for c in line if c.isdigit() or c == "."]))
                        nutrition[key.lower()] = value
                        break
                    except:
                        continue

        # Return all three values
        return text, nutrition, name
    except Exception as e:
        return f"Error generating recipe: {e}", {}


# ------------------------------------------------------------
# APP PAGES
# ------------------------------------------------------------
def main():
    st.title("🍳 Hungy")
    st.subheader("AI-Powered Recipe Generator with Gemini + Complementary AI")

    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🏠 Recipe Generator", "📚 Recipe History", "ℹ️ About"])

    if check_api_status():
        st.sidebar.success("✅ Gemini AI Connected")
    else:
        st.sidebar.warning("⚠️ Gemini AI Offline")

    if "api_calls_made" not in st.session_state:
        st.session_state.api_calls_made = 0

    if page == "🏠 Recipe Generator":
        recipe_generator_page()
    elif page == "📚 Recipe History":
        recipe_history_page()
    else:
        about_page()


def recipe_generator_page():
    st.header("🎯 Generate a Recipe")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("📸 Upload ingredient photo", type=["jpg", "png", "jpeg"])
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Ingredients", use_container_width=True)
            if st.button("🔍 Identify Ingredients"):
                with st.spinner("Analyzing with Gemini..."):
                    st.session_state.ingredients = identify_ingredients_from_image(image)
                    if st.session_state.ingredients:
                        st.success(", ".join(st.session_state.ingredients))
                    else:
                        st.warning("Couldn't identify ingredients.")
    with col2:
        st.subheader("✏️ Manual Ingredient Entry")

        # Initialize ingredients and input in session state
        if 'ingredients' not in st.session_state:
            st.session_state.ingredients = []
        if 'manual_ingredient' not in st.session_state:
            st.session_state.manual_ingredient = ""

        # Function to add ingredient
        def add_ingredient():
            ing = st.session_state.manual_ingredient.strip()
            if ing and ing not in st.session_state.ingredients:
                st.session_state.ingredients.append(ing)
                # st.success(f"Added: {ing}")
            # Clear input box after adding
            st.session_state.manual_ingredient = ""

        # Text input (press Enter to add)
        st.text_input(
            "Add ingredient manually:",
            key="manual_ingredient",
            placeholder="e.g., chicken breast, broccoli, rice",
            on_change=add_ingredient
        )

        col_add, col_clear = st.columns([1, 1])

        with col_add:
            st.button("➕ Add Ingredient", on_click=add_ingredient)

        with col_clear:
            if st.button("🗑️ Clear All"):
                st.session_state.ingredients = []
                st.rerun()

        # Display current ingredients
        if st.session_state.ingredients:
            st.write("**🥗 Current ingredients:**")
            for i, ingredient in enumerate(st.session_state.ingredients):
                col_a, col_b = st.columns([4, 1])
                col_a.write(f"• {ingredient}")
                if col_b.button("❌", key=f"remove_{i}", help=f"Remove {ingredient}"):
                    st.session_state.ingredients.remove(ingredient)
                    st.rerun()

    st.divider()

    dietary = st.selectbox("🥗 Dietary Restrictions", ["None", "Vegetarian", "Vegan", "Gluten-free", "Keto", "Low-carb", "Dairy-free"])
    cuisine = st.selectbox("🌍 Cuisine Type", ["Any", "Italian", "Asian", "Mexican", "Indian", "Mediterranean"])
    difficulty = st.selectbox("👨‍🍳 Difficulty", ["Easy", "Medium", "Hard"])

    if st.button("🍳 Generate Recipe", type="primary"):
        if not st.session_state.ingredients:
            st.warning("Please add ingredients first!")
            return

        with st.spinner("Generating recipe and nutrition with Gemini..."):
            recipe_text, nutrition, name = generate_recipe_and_nutrition(
                st.session_state.ingredients, dietary, cuisine, difficulty
            )

        st.subheader("📋 Recipe")
        st.write(recipe_text)

        if nutrition:
            st.subheader("📊 Gemini Nutrition Estimate")
            cols = st.columns(3)
            nutrients = list(nutrition.items())
            for i, (n, v) in enumerate(nutrients):
                cols[i % 3].metric(n.capitalize(), f"{v}")

        # Complementary AI
        st.divider()
        st.subheader("💚 Health Analysis")
        score = health_score(nutrition)
        st.metric("Health Score", f"{score}/100")
        if score > 80:
            st.success("🥦 Excellent nutritional balance!")
        elif score > 60:
            st.info("🍽️ Fairly healthy and balanced.")
        else:
            st.warning("⚠️ May be high in fats/sugars.")

        if "recipe_history" in st.session_state and len(st.session_state.recipe_history) > 1:
            st.subheader("🍳 You May Also Like (AI Recommender)")

            # Combine title + ingredients for vectorization
            past_recipes_text = [
                f"{r['name']} {' '.join(r['ingredients'])}" for r in st.session_state.recipe_history
            ]
            rec = RecipeRecommender()
            rec.fit(past_recipes_text)

            # Recommend based on the latest recipe
            recs = rec.recommend(len(past_recipes_text) - 1)

            for idx, sim in recs:
                title = st.session_state.recipe_history[idx]['name']
                st.write(f"• {title} ({sim*100:.1f}% match)")

        # save_recipe("Generated Recipe", recipe_text, st.session_state.ingredients, nutrition)
        save_recipe(name, recipe_text, st.session_state.ingredients, nutrition)
        st.success("✅ Recipe saved to history!")


def recipe_history_page():
    st.header("📚 Recipe History")
    if "recipe_history" not in st.session_state or not st.session_state.recipe_history:
        st.info("No recipes yet.")
        return
    for r in reversed(st.session_state.recipe_history):
        with st.expander(f"🍽️ {r['name']} - {r['date']}"):
            st.write(r["content"])
            if "nutrition" in r and r["nutrition"]:
                st.write("**Nutrition:**")
                st.json(r["nutrition"])


def save_recipe(name, content, ingredients, nutrition):
    if "recipe_history" not in st.session_state:
        st.session_state.recipe_history = []
    st.session_state.recipe_history.append({
        "name": name,
        "content": content,
        "ingredients": ingredients,
        "nutrition": nutrition,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


def about_page():
    st.header("ℹ️ About Hungy")
    st.write("""
    **Hungy** combines:
    - 🍳 *Gemini AI* — for recipe generation & nutrition estimation  
    - 🧠 *Complementary AI* — health scoring + recipe recommendations  
    - 📸 *Vision* — ingredient recognition from photos  

    ⚡ *All in one smart food assistant to inspire healthier cooking.*
    """)


if __name__ == "__main__":
    main()
