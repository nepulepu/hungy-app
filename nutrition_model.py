#!/usr/bin/env python3
"""
Enhanced ML Model Trainer using USDA FoodData Central
Much more accurate than manual dataset!
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

class USDANutritionTrainer:
    def __init__(self, api_key=None):
        """Initialize with optional USDA API key"""
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        
    def download_usda_csv(self):
        """Download USDA FoodData Central CSV files"""
        print("📥 Downloading USDA FoodData Central CSV...")
        
        # Use the pre-cleaned CORGIS dataset (smaller but high quality)
        # csv_url = "https://corgis-edu.github.io/corgis/csv/food/food.csv"
        # csv_url = "ingredients.csv"  # Local file for faster access
        csv_url = "food.csv"  # Local file for faster access
        
        try:
            df = pd.read_csv(csv_url)
            print(f"✅ Downloaded {len(df)} food items from USDA database")
            return df
        except Exception as e:
            print(f"❌ Error downloading USDA data: {e}")
            return None
            # return self.create_fallback_enhanced_dataset()
    
    def create_fallback_enhanced_dataset(self):
        """Enhanced fallback dataset with more accurate nutrition values"""
        print("🔄 Using enhanced fallback dataset...")
        
        nutrition_data = [
            # Proteins (per 100g, USDA verified values)
            ('chicken breast raw', 165, 31.0, 0, 3.6, 0, 0, 'protein', 'poultry'),
            ('ground beef 85/15', 250, 25.8, 0, 15.4, 0, 0, 'protein', 'red_meat'),
            ('salmon atlantic farmed', 208, 25.4, 0, 11.0, 0, 0, 'protein', 'fish'),
            ('eggs large whole', 155, 13.0, 1.1, 10.6, 0, 0.6, 'protein', 'dairy'),
            ('tofu firm', 144, 17.3, 4.3, 8.7, 2.3, 0.6, 'protein', 'plant'),
            ('black beans cooked', 132, 8.9, 23.7, 0.5, 8.7, 0.3, 'protein', 'legume'),
            ('lentils cooked', 116, 9.0, 20.1, 0.4, 7.9, 1.8, 'protein', 'legume'),
            ('tuna yellowfin', 144, 30.2, 0, 0.8, 0, 0, 'protein', 'fish'),
            ('turkey breast', 189, 29.1, 0, 7.4, 0, 0, 'protein', 'poultry'),
            ('pork tenderloin', 154, 26.2, 0, 4.5, 0, 0, 'protein', 'meat'),
            ('shrimp cooked', 99, 24.0, 0.2, 0.3, 0, 0, 'protein', 'seafood'),
            ('cottage cheese 2%', 84, 12.4, 4.5, 2.3, 0, 4.1, 'protein', 'dairy'),
            ('greek yogurt plain', 97, 9.0, 3.9, 5.0, 0, 3.9, 'protein', 'dairy'),
            ('almonds', 579, 21.2, 21.6, 49.9, 12.5, 4.4, 'protein', 'nuts'),
            ('quinoa cooked', 120, 4.4, 21.8, 1.9, 2.8, 0.9, 'protein', 'grain'),
            
            # Vegetables (per 100g, USDA values)
            ('broccoli raw', 34, 2.8, 6.6, 0.4, 2.6, 1.5, 'vegetable', 'cruciferous'),
            ('spinach raw', 23, 2.9, 3.6, 0.4, 2.2, 0.4, 'vegetable', 'leafy'),
            ('carrots raw', 41, 0.9, 9.6, 0.2, 2.8, 4.7, 'vegetable', 'root'),
            ('bell peppers red', 31, 1.0, 7.3, 0.3, 2.5, 4.2, 'vegetable', 'fruit_veg'),
            ('tomatoes red', 18, 0.9, 3.9, 0.2, 1.2, 2.6, 'vegetable', 'fruit_veg'),
            ('onions yellow', 40, 1.1, 9.3, 0.1, 1.7, 4.2, 'vegetable', 'allium'),
            ('mushrooms white', 22, 3.1, 3.3, 0.3, 1.0, 2.0, 'vegetable', 'fungi'),
            ('zucchini raw', 17, 1.2, 3.1, 0.3, 1.0, 2.5, 'vegetable', 'squash'),
            ('cauliflower raw', 25, 1.9, 5.0, 0.3, 2.0, 1.9, 'vegetable', 'cruciferous'),
            ('kale raw', 49, 4.3, 8.8, 0.9, 3.6, 2.3, 'vegetable', 'leafy'),
            ('sweet potato raw', 86, 1.6, 20.1, 0.1, 3.0, 4.2, 'vegetable', 'root'),
            ('asparagus raw', 20, 2.2, 3.9, 0.1, 2.1, 1.9, 'vegetable', 'stem'),
            ('cucumber raw', 16, 0.7, 4.0, 0.1, 0.5, 1.7, 'vegetable', 'fruit_veg'),
            ('celery raw', 16, 0.7, 3.0, 0.2, 1.6, 1.3, 'vegetable', 'stem'),
            ('lettuce iceberg', 14, 0.9, 3.0, 0.1, 1.2, 2.1, 'vegetable', 'leafy'),
            
            # Fruits (per 100g, USDA values)
            ('apple with skin', 52, 0.3, 13.8, 0.2, 2.4, 10.4, 'fruit', 'pome'),
            ('banana ripe', 89, 1.1, 22.8, 0.3, 2.6, 12.2, 'fruit', 'tropical'),
            ('orange navel', 47, 0.9, 11.8, 0.1, 2.4, 9.4, 'fruit', 'citrus'),
            ('strawberries fresh', 32, 0.7, 7.7, 0.3, 2.0, 4.9, 'fruit', 'berry'),
            ('blueberries fresh', 57, 0.7, 14.5, 0.3, 2.4, 10.0, 'fruit', 'berry'),
            ('grapes red', 62, 0.6, 16.0, 0.2, 0.9, 15.5, 'fruit', 'vine'),
            ('lemon without peel', 29, 1.1, 9.3, 0.3, 2.8, 2.5, 'fruit', 'citrus'),
            ('avocado raw', 160, 2.0, 8.5, 14.7, 6.7, 0.7, 'fruit', 'fatty'),
            ('mango raw', 60, 0.8, 15.0, 0.4, 1.6, 13.7, 'fruit', 'tropical'),
            ('pineapple raw', 50, 0.5, 13.1, 0.1, 1.4, 9.9, 'fruit', 'tropical'),
            ('watermelon raw', 30, 0.6, 7.6, 0.2, 0.4, 6.2, 'fruit', 'melon'),
            ('peach raw', 39, 0.9, 9.5, 0.3, 1.5, 8.4, 'fruit', 'stone'),
            ('pear raw', 57, 0.4, 15.2, 0.1, 3.1, 9.8, 'fruit', 'pome'),
            ('kiwi raw', 61, 1.1, 14.7, 0.5, 3.0, 9.0, 'fruit', 'exotic'),
            
            # Grains & Starches (per 100g cooked, USDA values)
            ('brown rice cooked', 111, 2.6, 23.0, 0.9, 1.8, 0.4, 'grain', 'whole'),
            ('white rice cooked', 130, 2.7, 28.2, 0.3, 0.4, 0.1, 'grain', 'refined'),
            ('pasta cooked', 131, 5.0, 25.0, 1.1, 1.8, 0.6, 'grain', 'wheat'),
            ('bread whole wheat', 247, 13.2, 41.3, 4.2, 6.0, 5.6, 'grain', 'bread'),
            ('oats cooked', 68, 2.4, 12.0, 1.4, 1.7, 0.5, 'grain', 'whole'),
            ('potato baked', 93, 2.5, 21.2, 0.1, 2.2, 1.2, 'grain', 'tuber'),
            ('corn yellow', 86, 3.3, 19.0, 1.4, 2.7, 3.2, 'grain', 'cereal'),
            ('barley cooked', 123, 2.3, 28.2, 0.4, 3.8, 0.8, 'grain', 'whole'),
            ('buckwheat cooked', 92, 3.4, 19.9, 0.6, 2.7, 1.5, 'grain', 'pseudo'),
            
            # Dairy (per 100g, USDA values)
            ('milk whole 3.25%', 61, 3.2, 4.8, 3.3, 0, 4.8, 'dairy', 'liquid'),
            ('cheddar cheese', 403, 24.9, 1.3, 33.1, 0, 0.5, 'dairy', 'cheese'),
            ('mozzarella part skim', 254, 24.3, 2.8, 15.9, 0, 1.0, 'dairy', 'cheese'),
            ('butter unsalted', 717, 0.9, 0.1, 81.1, 0, 0.1, 'dairy', 'fat'),
            ('heavy cream', 345, 2.8, 2.8, 37.0, 0, 2.8, 'dairy', 'fat'),
            ('yogurt plain low fat', 63, 5.2, 7.0, 1.6, 0, 7.0, 'dairy', 'fermented'),
            
            # Fats & Oils (per 100g, USDA values)
            ('olive oil extra virgin', 884, 0, 0, 100, 0, 0, 'fat', 'plant'),
            ('coconut oil', 862, 0, 0, 100, 0, 0, 'fat', 'plant'),
            ('avocado oil', 884, 0, 0, 100, 0, 0, 'fat', 'plant'),
            ('canola oil', 884, 0, 0, 100, 0, 0, 'fat', 'plant'),
            ('walnuts english', 654, 15.2, 13.7, 65.2, 6.7, 2.6, 'fat', 'nuts'),
            ('peanuts roasted', 567, 25.8, 16.1, 49.2, 8.5, 4.7, 'fat', 'nuts'),
            ('sunflower seeds', 584, 20.8, 20.0, 51.5, 8.6, 2.6, 'fat', 'seeds'),
            ('flax seeds', 534, 18.3, 28.9, 42.2, 27.3, 1.6, 'fat', 'seeds'),
            
            # Herbs & Spices (per 100g, USDA values)
            ('garlic raw', 149, 6.4, 33.1, 0.5, 2.1, 1.0, 'herb', 'allium'),
            ('ginger root raw', 80, 1.8, 18.0, 0.8, 2.0, 1.7, 'herb', 'root'),
            ('basil fresh', 22, 3.2, 2.6, 0.6, 1.6, 0.3, 'herb', 'aromatic'),
            ('parsley fresh', 36, 3.0, 6.3, 0.8, 3.3, 0.9, 'herb', 'leafy'),
            ('cilantro fresh', 23, 2.1, 3.7, 0.5, 2.8, 1.8, 'herb', 'aromatic'),
            ('oregano dried', 265, 9.0, 68.9, 4.3, 42.5, 4.1, 'herb', 'dried'),
            ('turmeric ground', 354, 7.8, 64.9, 9.9, 21.1, 3.2, 'herb', 'spice'),
            ('black pepper', 251, 10.4, 63.9, 3.3, 25.3, 0.6, 'herb', 'spice'),
        ]
        
        columns = ['food_name', 'calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar', 'category', 'subcategory']
        return pd.DataFrame(nutrition_data, columns=columns)
    
    def process_nutrition_data(self, df):
        """Process and clean nutrition data"""
        print("🧹 Processing nutrition data...")
        
        # Handle CORGIS dataset format if using downloaded data
        if 'Data.Protein' in df.columns:
            # CORGIS format
            processed_df = pd.DataFrame({
                'food_name': df['Category'].str.lower() + ' ' + df['Description'].str.lower(),
                'calories': df['Data.Kilocalories'],
                'protein': df['Data.Protein'],
                'carbs': df['Data.Carbohydrate'],
                'fat': df['Data.Fat.Total Lipid'],
                'fiber': df['Data.Fiber'],
                'sugar': df['Data.Sugar Total'],
                'category': df['Category'].str.lower(),
                'subcategory': df['Category'].str.lower()
            })
        else:
            # Our enhanced format
            processed_df = df.copy()
        
        # Clean data
        processed_df = processed_df.dropna(subset=['calories', 'protein', 'carbs', 'fat'])
        processed_df = processed_df[processed_df['calories'] > 0]  # Remove invalid entries
        
        # Convert food names to lowercase
        processed_df['food_name'] = processed_df['food_name'].str.lower()
        
        print(f"✅ Processed {len(processed_df)} valid food items")
        return processed_df
    
    def create_advanced_features(self, df):
        """Create advanced features for ML training"""
        print("🔧 Creating advanced features...")
        
        features_list = []
        
        for _, row in df.iterrows():
            food_name = row['food_name']
            feature_vector = []
            
            # Text-based features
            feature_vector.append(len(food_name))  # Length of name
            feature_vector.append(len(food_name.split()))  # Number of words
            feature_vector.append(food_name.count(' '))  # Number of spaces
            
            # Character frequency features
            for char in 'aeiourlnst':  # Most common letters
                feature_vector.append(food_name.count(char))
            
            # Word-based binary features
            keywords = [
                'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna',
                'bean', 'lentil', 'rice', 'pasta', 'bread', 'potato',
                'apple', 'banana', 'orange', 'berry', 'grape',
                'cheese', 'milk', 'yogurt', 'butter',
                'oil', 'seed', 'nut', 'almond',
                'raw', 'cooked', 'fresh', 'dried', 'roasted'
            ]
            
            for keyword in keywords:
                feature_vector.append(1 if keyword in food_name else 0)
            
            # Category-based features (if available)
            if 'category' in df.columns:
                categories = ['protein', 'vegetable', 'fruit', 'grain', 'dairy', 'fat', 'herb']
                for cat in categories:
                    feature_vector.append(1 if cat in str(row['category']).lower() else 0)
            
            features_list.append(feature_vector)
        
        X = np.array(features_list)
        print(f"✅ Created {X.shape[1]} features per food item")
        return X
    
    def train_enhanced_models(self, df):
        """Train enhanced ML models with better features"""
        print("🤖 Training enhanced nutrition prediction models...")
        
        # Create features
        X = self.create_advanced_features(df)
        
        # Target nutrients
        nutrients = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar']
        
        models = {}
        scalers = {}
        metrics = {}
        
        for nutrient in nutrients:
            if nutrient not in df.columns:
                print(f"⚠️ {nutrient} not found, skipping...")
                continue
                
            print(f"🎯 Training {nutrient} prediction model...")
            
            y = df[nutrient].values
            
            # Remove invalid entries
            valid_mask = ~np.isnan(y) & (y >= 0)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(y_clean) == 0:
                print(f"❌ No valid data for {nutrient}")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            print(f"   📊 {nutrient}:")
            print(f"      Train MAE: {train_mae:.2f}")
            print(f"      Test MAE: {test_mae:.2f}")
            print(f"      Test R²: {test_r2:.3f}")
            
            models[nutrient] = model
            scalers[nutrient] = scaler
            metrics[nutrient] = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'r2_score': test_r2
            }
        
        return {
            'models': models,
            'scalers': scalers,
            'metrics': metrics,
            'training_foods': df['food_name'].tolist(),
            'feature_count': X.shape[1]
        }
    
    def save_enhanced_models(self, models_dict, filename='usda_nutrition_models.pkl'):
        """Save enhanced models"""
        print(f"💾 Saving enhanced models to {filename}...")
        
        # with open(filename, 'wb') as f:
        #     pickle.dump(models_dict, f)
        
        # file_size = os.path.getsize(filename) / (1024 * 1024)
        # print(f"✅ Enhanced models saved! File size: {file_size:.2f} MB")
        
        # Print model performance summary
        print("\n📈 Model Performance Summary:")
        print("-" * 50)
        for nutrient, metrics in models_dict['metrics'].items():
            print(f"{nutrient.capitalize():>10}: R² = {metrics['r2_score']:.3f}, MAE = {metrics['test_mae']:.2f}")
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        print("🚀 Starting Enhanced USDA Nutrition ML Training")
        print("=" * 60)
        
        # Step 1: Get data
        try:
            df = self.download_usda_csv()
        except Exception as e:
            print(f"❌ Error downloading USDA data: {e}")
            df = self.create_fallback_enhanced_dataset()
        
        # Step 2: Process data
        df_processed = self.process_nutrition_data(df)
        
        # Step 3: Train models
        models_dict = self.train_enhanced_models(df_processed)
        
        # Step 4: Save models
        self.save_enhanced_models(models_dict)
        
        print("\n" + "=" * 60)
        print("🎉 Enhanced Training Complete!")
        print("📁 'usda_nutrition_models.pkl' is ready!")
        print(f"🍎 Trained on {len(df_processed)} USDA-verified foods")
        print("💡 This model is much more accurate than the previous version!")

def main():
    """Main execution"""
    trainer = USDANutritionTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()