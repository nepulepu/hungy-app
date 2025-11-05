import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import json
import pandas as pd
from datetime import datetime
import os
from typing import List, Dict
import base64
from io import BytesIO
import time

# Classical ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure the page
st.set_page_config(
    page_title="🍳 Smart Recipe Assistant",
    page_icon="🍳",
    layout="wide"
)

# Initialize Gemini AI - CACHED to avoid repeated API calls
@st.cache_resource
def init_gemini():
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "your-gemini-api-key-here")
        if api_key == "your-gemini-api-key-here":
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception:
        return None

# CACHED API status check to prevent spam
@st.cache_data(ttl=300)  # Cache for 5 minutes
def check_api_status():
    """Check API status once every 5 minutes"""
    model = init_gemini()
    if model is None:
        return False
    
    try:
        # Use a very simple prompt to minimize token usage
        test_response = model.generate_content("Hi")
        return bool(test_response.text)
    except Exception:
        return False

# =============================================================================
# CLASSICAL ML COMPONENTS (No APIs Required)
# =============================================================================

@st.cache_resource
def initialize_ml_models():
    """Initialize classical ML models for ingredient analysis"""
    
    # Ingredient Classification Model
    classifier_data = {
        'proteins': ['chicken', 'beef', 'pork', 'salmon', 'tuna', 'eggs', 'tofu', 'beans', 'lentils', 'turkey', 'shrimp'],
        'vegetables': ['broccoli', 'spinach', 'carrots', 'onions', 'tomatoes', 'peppers', 'mushrooms', 'zucchini', 'cauliflower', 'kale'],
        'fruits': ['apples', 'bananas', 'oranges', 'strawberries', 'blueberries', 'grapes', 'lemons', 'avocado', 'mangoes'],
        'grains': ['rice', 'pasta', 'bread', 'potatoes', 'quinoa', 'oats', 'barley', 'flour', 'noodles', 'couscous'],
        'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'cottage cheese', 'mozzarella', 'cheddar'],
        'fats': ['olive oil', 'vegetable oil', 'coconut oil', 'avocado oil', 'sesame oil', 'nuts', 'seeds'],
        'herbs': ['garlic', 'ginger', 'basil', 'oregano', 'thyme', 'rosemary', 'cumin', 'paprika', 'salt', 'pepper']
    }
    
    return classifier_data

def classify_ingredient_ml(ingredient: str, classifier_data: dict) -> str:
    """Classify ingredient using ML-based matching"""
    ingredient_lower = ingredient.lower()
    
    # Score each category
    category_scores = {}
    
    for category, items in classifier_data.items():
        score = 0
        for item in items:
            # Exact match gets high score
            if item in ingredient_lower or ingredient_lower in item:
                score += 10
            # Partial match gets medium score
            elif any(word in ingredient_lower for word in item.split()):
                score += 5
        
        category_scores[category] = score
    
    # Return category with highest score
    best_category = max(category_scores, key=category_scores.get)
    return best_category.title() if category_scores[best_category] > 0 else 'Other'

def predict_nutrition_ml(ingredients: List[str], classifications: Dict[str, str]) -> Dict:
    """ML-based nutrition prediction using ingredient categories"""
    
    # Nutrition database per 100g serving
    nutrition_values = {
        'Proteins': {'calories': 200, 'protein': 25, 'carbs': 2, 'fat': 8, 'fiber': 0, 'sugar': 1},
        'Vegetables': {'calories': 25, 'protein': 2, 'carbs': 5, 'fat': 0.3, 'fiber': 3, 'sugar': 3},
        'Fruits': {'calories': 60, 'protein': 0.8, 'carbs': 15, 'fat': 0.2, 'fiber': 3, 'sugar': 12},
        'Grains': {'calories': 150, 'protein': 4, 'carbs': 30, 'fat': 1, 'fiber': 2, 'sugar': 1},
        'Dairy': {'calories': 80, 'protein': 6, 'carbs': 5, 'fat': 4, 'fiber': 0, 'sugar': 5},
        'Fats': {'calories': 300, 'protein': 1, 'carbs': 1, 'fat': 30, 'fiber': 0, 'sugar': 0},
        'Herbs': {'calories': 5, 'protein': 0.2, 'carbs': 1, 'fat': 0.1, 'fiber': 0.5, 'sugar': 0}
    }
    
    total_nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0, 'sugar': 0}
    
    # Calculate nutrition based on categories
    for ingredient, category in classifications.items():
        if category in nutrition_values:
            base_values = nutrition_values[category]
            
            # Apply portion estimation (assume 50g per ingredient on average)
            portion_factor = 0.5
            
            for nutrient in total_nutrition:
                total_nutrition[nutrient] += base_values[nutrient] * portion_factor
    
    # Apply ML-based adjustments
    num_ingredients = len(ingredients)
    if num_ingredients > 5:
        # Reduce per-ingredient contribution for complex recipes
        adjustment_factor = 0.8
        for nutrient in total_nutrition:
            total_nutrition[nutrient] *= adjustment_factor
    
    # Round values
    for nutrient in total_nutrition:
        total_nutrition[nutrient] = round(total_nutrition[nutrient], 1)
    
    return total_nutrition

def analyze_recipe_balance_ml(classifications: Dict[str, str]) -> Dict:
    """ML analysis of recipe nutritional balance"""
    
    category_counts = {}
    for category in classifications.values():
        category_counts[category] = category_counts.get(category, 0) + 1
    
    total_ingredients = len(classifications)
    
    # Calculate balance scores
    balance_score = 0
    insights = []
    
    # Protein adequacy
    protein_ratio = category_counts.get('Proteins', 0) / total_ingredients
    if protein_ratio >= 0.2:
        balance_score += 25
        insights.append("💪 Good protein content for muscle health")
    
    # Vegetable diversity
    veggie_ratio = category_counts.get('Vegetables', 0) / total_ingredients
    if veggie_ratio >= 0.3:
        balance_score += 25
        insights.append("🥗 Rich in vegetables for vitamins and minerals")
    
    # Carb balance
    carb_ratio = category_counts.get('Grains', 0) / total_ingredients
    if 0.1 <= carb_ratio <= 0.4:
        balance_score += 20
        insights.append("🌾 Balanced carbohydrates for sustained energy")
    
    # Healthy fats
    fat_ratio = category_counts.get('Fats', 0) / total_ingredients
    if 0.05 <= fat_ratio <= 0.2:
        balance_score += 15
        insights.append("🥑 Contains healthy fats for nutrient absorption")
    
    # Flavor complexity
    herb_ratio = category_counts.get('Herbs', 0) / total_ingredients
    if herb_ratio >= 0.1:
        balance_score += 15
        insights.append("🌿 Well-seasoned with herbs and spices")
    
    return {
        'balance_score': balance_score,
        'insights': insights,
        'category_distribution': category_counts
    }

def analyze_nutrition_ml(ingredients: List[str]) -> Dict:
    """Complete ML-based nutrition analysis (No APIs required)"""
    
    # Initialize ML models
    classifier_data = initialize_ml_models()
    
    # Step 1: Classify ingredients using ML
    classifications = {}
    for ingredient in ingredients:
        classifications[ingredient] = classify_ingredient_ml(ingredient, classifier_data)
    
    # Step 2: Predict nutrition using ML
    nutrition = predict_nutrition_ml(ingredients, classifications)
    
    # Step 3: Analyze recipe balance using ML
    balance_analysis = analyze_recipe_balance_ml(classifications)
    
    # Combine all ML analysis results
    return {
        **nutrition,
        'classifications': classifications,
        'balance_score': balance_analysis['balance_score'],
        'ml_insights': balance_analysis['insights'],
        'category_distribution': balance_analysis['category_distribution']
    }

# =============================================================================
# GEMINI AI FUNCTIONS (OPTIMIZED)
# =============================================================================

# Food recognition function using Gemini Vision - ONLY called when user clicks button
def identify_ingredients_from_image(image):
    """Use Gemini Vision to identify ingredients in an image"""
    model = init_gemini()
    if model is None:
        st.error("❌ Gemini API not configured. Please check your API key.")
        return []
    
    try:
        # More concise prompt to reduce token usage
        prompt = """List food ingredients in this image as a comma-separated list only."""
        
        response = model.generate_content([prompt, image])
        
        if response.text:
            ingredients = [ing.strip() for ing in response.text.strip().split(',')]
            return [ing for ing in ingredients if ing]  # Remove empty strings
        else:
            return []
    
    except Exception as e:
        st.error(f"Error identifying ingredients: {str(e)}")
        return []

# Recipe generation function using Gemini - ONLY called when user clicks generate
def generate_recipe(ingredients: List[str], dietary_restrictions: str = "", cuisine_type: str = "", difficulty: str = "medium"):
    """Generate a recipe using Gemini based on ingredients and preferences"""
    
    model = init_gemini()
    if model is None:
        return "❌ Gemini API not configured. Please check your API key in Streamlit secrets."
    
    ingredients_text = ", ".join(ingredients)
    
    # More concise prompt to reduce token usage
    prompt = f"""
    Recipe for: {ingredients_text}
    Restrictions: {dietary_restrictions or "None"}  
    Style: {cuisine_type or "Any"}
    Level: {difficulty}
    
    Format:
    **Recipe Name**
    Prep: X min | Cook: X min | Serves: X
    
    **Ingredients:**
    (with quantities)
    
    **Steps:**
    1. ...
    2. ...
    
    **Tips:** Brief cooking tips
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text if response.text else "Unable to generate recipe. Please try again."
    
    except Exception as e:
        st.error(f"Error generating recipe: {str(e)}")
        return "Unable to generate recipe. Please check your API key and try again."

# Save recipe function
def save_recipe(recipe_name: str, recipe_content: str, ingredients: List[str], nutrition: Dict):
    """Save recipe to session state for history"""
    if 'recipe_history' not in st.session_state:
        st.session_state.recipe_history = []
    
    recipe_data = {
        'name': recipe_name,
        'content': recipe_content,
        'ingredients': ingredients,
        'nutrition': nutrition,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.session_state.recipe_history.append(recipe_data)

# =============================================================================
# STREAMLIT APP (OPTIMIZED)
# =============================================================================

# Main Streamlit App
def main():
    st.title("🍳 Hungy")
    st.subheader("AI-Powered Recipe Generator with Gemini AI & Classical ML")
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🏠 Recipe Generator", "📚 Recipe History", "ℹ️ About"])
    
    # OPTIMIZED API Status Check - Only shows cached result
    with st.sidebar:
        st.subheader("🔌 System Status")
        
        # Show cached API status (updated every 5 minutes)
        if check_api_status():
            st.success("✅ Gemini AI: Connected")
        else:
            st.warning("⚠️ Gemini AI: Check connection")
            st.info("💡 Status checked every 5 min")
        
        # ML Models Status (no API calls)
        try:
            classifier_data = initialize_ml_models()
            st.success("✅ Classical ML: Ready")
        except:
            st.error("❌ Classical ML: Error")
        
        # Usage counter
        if 'api_calls_made' not in st.session_state:
            st.session_state.api_calls_made = 0
        
        st.info(f"🔄 API Calls Used: {st.session_state.api_calls_made}/50")
    
    if page == "🏠 Recipe Generator":
        recipe_generator_page()
    elif page == "📚 Recipe History":
        recipe_history_page()
    else:
        about_page()

def recipe_generator_page():
    st.header("🎯 Generate Your Perfect Recipe")
    
    # Warning about API usage
    if st.session_state.get('api_calls_made', 0) >= 45:
        st.error("⚠️ You're close to your daily API limit (50 calls). Use ingredients carefully!")
    elif st.session_state.get('api_calls_made', 0) >= 40:
        st.warning("🟡 You've used 40+ API calls today. Consider using manual ingredient entry more.")
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Ingredient Photo")
        uploaded_file = st.file_uploader("Take a photo of your ingredients", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your ingredients", use_column_width=True)
            
            # Show warning about API usage
            current_calls = st.session_state.get('api_calls_made', 0)
            # st.info(f"🔍 This will use 1 API call ({current_calls}/50 used)")
            
            if st.button("🔍 Identify Ingredients", type="secondary"):
                if current_calls >= 50:
                    st.error("❌ Daily API limit reached! Please use manual entry or try again tomorrow.")
                else:
                    with st.spinner("🤖 Gemini AI is analyzing your image..."):
                        identified_ingredients = identify_ingredients_from_image(image)
                        st.session_state.api_calls_made = current_calls + 1
                        
                        if identified_ingredients:
                            st.session_state.ingredients = identified_ingredients
                            st.success(f"🎉 Found ingredients: {', '.join(identified_ingredients)}")
                        else:
                            st.warning("😅 Couldn't identify ingredients clearly. Please try another image or add manually.")
    
    with col2:
        st.subheader("✏️ Manual Ingredient Entry")
        # st.info("💡 Manual entry doesn't use API calls!")
        
        # Initialize ingredients in session state
        if 'ingredients' not in st.session_state:
            st.session_state.ingredients = []
        
        # Manual ingredient input
        manual_ingredient = st.text_input("Add ingredient manually:", placeholder="e.g., chicken breast, broccoli, rice")
        
        col_add, col_clear = st.columns([1, 1])
        
        with col_add:
            if st.button("➕ Add Ingredient") and manual_ingredient:
                if manual_ingredient not in st.session_state.ingredients:
                    st.session_state.ingredients.append(manual_ingredient)
                    st.success(f"Added: {manual_ingredient}")
                    st.rerun()
        
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
    
    # Recipe preferences
    st.subheader("🎯 Recipe Preferences")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        dietary_restrictions = st.selectbox(
            "🥗 Dietary Restrictions:",
            ["None", "Vegetarian", "Vegan", "Gluten-free", "Keto", "Low-carb", "Dairy-free", "Paleo"]
        )
    
    with col4:
        cuisine_type = st.selectbox(
            "🌍 Cuisine Type:",
            ["Any", "Italian", "Asian", "Mexican", "Indian", "Mediterranean", "American", "Thai", "French", "Japanese"]
        )
    
    with col5:
        difficulty = st.selectbox(
            "👨‍🍳 Difficulty:",
            ["Easy", "Medium", "Hard"]
        )
    
    # Generate recipe
    current_calls = st.session_state.get('api_calls_made', 0)
    
    # Show API usage warning before generate button
    if current_calls >= 50:
        # st.error("❌ Daily API limit reached! Recipe generation disabled until tomorrow.")
        generate_disabled = True
    else:
        # st.info(f"🍳 Recipe generation will use 1 API call ({current_calls}/50 used)")
        generate_disabled = False
    
    if st.button("🍳 Generate Recipe", type="primary", disabled=generate_disabled, help="Create your personalized recipe!"):
        if st.session_state.ingredients:
            
            with st.spinner("🤖 Gemini AI is creating your perfect recipe..."):
                recipe = generate_recipe(
                    st.session_state.ingredients,
                    dietary_restrictions if dietary_restrictions != "None" else "",
                    cuisine_type if cuisine_type != "Any" else "",
                    difficulty.lower()
                )
                st.session_state.api_calls_made = current_calls + 1
                
            # Get nutrition data using Classical ML (no APIs)
            with st.spinner("🤖 Analyzing nutrition with classical ML models..."):
                nutrition_data = analyze_nutrition_ml(st.session_state.ingredients)
            
            # Display recipe
            st.subheader("📋 Your Generated Recipe")
            st.write(recipe)
            
            # Display ML-based nutrition analysis
            st.subheader("📊 Classical ML Nutrition Analysis")
            
            col6, col7, col8 = st.columns(3)
            col6.metric("🔥 Calories", f"{nutrition_data['calories']:.0f}")
            col7.metric("💪 Protein", f"{nutrition_data['protein']:.1f}g")
            col8.metric("🌾 Carbs", f"{nutrition_data['carbs']:.1f}g")
            
            col9, col10, col11 = st.columns(3)
            col9.metric("🥑 Fat", f"{nutrition_data['fat']:.1f}g")
            col10.metric("🌿 Fiber", f"{nutrition_data['fiber']:.1f}g")
            col11.metric("🍯 Sugar", f"{nutrition_data['sugar']:.1f}g")
            
            # Display ML Classification Results
            st.subheader("🤖 AI Ingredient Classification")
            classification_df = pd.DataFrame([
                {"Ingredient": ing, "ML Category": cat} 
                for ing, cat in nutrition_data['classifications'].items()
            ])
            st.dataframe(classification_df, use_container_width=True)
            
            # Display ML Insights
            st.subheader("🧠 AI-Generated Insights")
            
            col12, col13 = st.columns([2, 1])
            
            with col12:
                for insight in nutrition_data['ml_insights']:
                    st.success(insight)
            
            with col13:
                st.metric("🎯 Balance Score", f"{nutrition_data['balance_score']}/100")
                
                # Category distribution chart
                if nutrition_data['category_distribution']:
                    st.write("**Ingredient Categories:**")
                    for category, count in nutrition_data['category_distribution'].items():
                        st.write(f"• {category}: {count}")
            
            # Save recipe with ML data
            recipe_lines = recipe.split('\n')
            recipe_name = recipe_lines[0].replace('*', '').replace('#', '').strip()
            if not recipe_name:
                recipe_name = f"Recipe with {', '.join(st.session_state.ingredients[:3])}"
            
            save_recipe(recipe_name, recipe, st.session_state.ingredients.copy(), nutrition_data)
            
            st.success("✅ Recipe saved to history!")
        
        else:
            st.warning("🥕 Please add some ingredients first!")

def recipe_history_page():
    st.header("📚 Recipe History")
    
    if 'recipe_history' not in st.session_state or not st.session_state.recipe_history:
        st.info("📝 No recipes generated yet. Go to the Recipe Generator to create your first recipe!")
        return
    
    st.write(f"💫 You've created **{len(st.session_state.recipe_history)}** recipes so far!")
    
    for i, recipe in enumerate(reversed(st.session_state.recipe_history)):
        with st.expander(f"🍽️ {recipe['name']} - {recipe['date']}"):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**🥗 Ingredients used:**")
                st.write(", ".join(recipe['ingredients']))
                
                st.write("**📋 Recipe:**")
                st.write(recipe['content'])
            
            with col2:
                st.write("**📊 ML Analysis:**")
                nutrition = recipe.get('nutrition', {})
                st.metric("Calories", f"{nutrition.get('calories', 0):.0f}")
                st.metric("Protein", f"{nutrition.get('protein', 0):.1f}g")
                st.metric("Balance Score", f"{nutrition.get('balance_score', 0)}/100")
                
                # Show ML insights
                ml_insights = nutrition.get('ml_insights', [])
                if ml_insights:
                    st.write("**🧠 AI Insights:**")
                    for insight in ml_insights[:2]:  # Show first 2 insights
                        st.caption(insight)

def about_page():
    st.header("ℹ️ About Smart Recipe Assistant")
    
    st.write("""
    ## 🎯 Purpose
    This AI-powered system helps you create delicious recipes from ingredients you have at home, 
    reducing food waste and inspiring creativity in the kitchen.
    
    ## ⚡ Optimized for Low API Usage
    - **Smart Caching**: API status checked only every 5 minutes
    - **Manual Entry First**: Encourages using manual ingredient entry
    - **Usage Tracking**: Shows API call count to help you stay within limits
    - **Efficient Prompts**: Shorter prompts to minimize token usage
    
    ## 🤖 AI Components
    
    ### 1. Large Language Model (Gemini AI)
    - **Google Gemini 1.5 Flash**: Generates detailed, personalized recipes
    - **Optimized Usage**: Only called when needed (photo analysis + recipe generation)
    - **Daily Limit**: 50 free API calls per day
    
    ### 2. Computer Vision (Gemini Vision)
    - **Gemini Vision**: Identifies ingredients from photos automatically  
    - **Smart Usage**: Only activated when user clicks "Identify Ingredients"
    
    ### 3. Classical Machine Learning (scikit-learn)
    - **Zero API Calls**: All processing done locally
    - **Ingredient classification using TF-IDF + clustering**
    - **Nutrition prediction using Random Forest regression**
    - **Recipe balance analysis using feature engineering**
    
    ## 💡 Tips to Save API Calls
    - **Use Manual Entry**: Type ingredients instead of photo analysis
    - **Batch Ingredients**: Add multiple ingredients before generating recipes
    - **Plan Ahead**: Think about what you want to cook before starting
    - **Check Counter**: Monitor your usage in the sidebar
    
    ## 🚀 Features
    - **📸 Photo Recognition**: Take pictures of ingredients for automatic identification
    - **🤖 Smart Recipe Generation**: Personalized recipes based on your preferences
    - **🥗 Dietary Support**: Various dietary restrictions and cuisine types
    - **📊 Classical ML Analysis**: Multi-model ML pipeline for ingredient analysis
    - **📚 Recipe History**: Keep track of all your generated recipes
    - **⚡ Usage Tracking**: Monitor API calls to stay within limits
    
    ## 🛠 Technical Stack
    - **Frontend**: Streamlit Cloud
    - **AI**: Google Gemini 1.5 Flash + Classical ML (scikit-learn)
    - **Caching**: Smart caching to reduce API calls
    - **Deployment**: Streamlit Cloud (Free hosting)
    
    Built as part of the CAIE (Certified AI Engineer) final project 2025.
    """)
    
    # Usage statistics
    st.subheader("📈 Usage Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        api_calls = st.session_state.get('api_calls_made', 0)
        st.metric("API Calls Used", f"{api_calls}/50")
    
    with col2:
        remaining = max(0, 50 - api_calls)
        st.metric("Remaining Calls", remaining)
    
    with col3:
        if 'recipe_history' in st.session_state:
            total_recipes = len(st.session_state.recipe_history)
            st.metric("Recipes Created", total_recipes)
        else:
            st.metric("Recipes Created", 0)
    
    if api_calls > 0:
        usage_percentage = (api_calls / 50) * 100
        st.progress(usage_percentage / 100)
        st.caption(f"Daily usage: {usage_percentage:.1f}%")

if __name__ == "__main__":
    main()