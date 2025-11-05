# 🍳 Hungy

Hungy is an AI-powered smart food assistant that helps you generate personalized recipes based on ingredients, dietary preferences, and cuisine styles. It leverages **Gemini AI** for recipe generation, nutrition estimation, and visual ingredient recognition — combined with a **Complementary AI** module for health scoring and smart recipe recommendations.

---

## 📑 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

---

## 🚀 Features

- 📸 **Ingredient Recognition from Photos** using Gemini Vision
- ✍️ **Manual Ingredient Entry** via user-friendly input
- 🍲 **Recipe Generation** (with steps, ingredients, and cook time)
- 📊 **Nutrition Estimation** (per serving, including macros)
- 💚 **Health Score** calculation based on nutritional balance
- 🤖 **AI-Based Recipe Recommendations** (TF-IDF + cosine similarity)
- 🧠 **Session-based Recipe History** for tracking past creations
- 🌍 **Custom Cuisine & Dietary Filters** (e.g., vegan, keto, etc.)

---

## 💾 Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/hungy.git
   cd hungy
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   - Create a `.streamlit/secrets.toml` file:
     ```toml
     GEMINI_API_KEY = "your-gemini-api-key-here"
     ```

---

## 🧑‍🍳 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Navigate to:
```
http://localhost:8501
```

### Interface Sections

- 🏠 **Recipe Generator**: Upload photo or manually add ingredients, set preferences, and generate.
- 📚 **Recipe History**: View and revisit previously generated recipes.
- ℹ️ **About**: Learn more about the app’s capabilities.

---

## 📦 Dependencies

From `requirements.txt`:

- `streamlit==1.49.1`
- `google-generativeai==0.3.2`
- `Pillow==11.3.0`
- `pandas==2.3.2`
- `requests==2.31.0`
- `python-dotenv==1.0.0`
- `scikit-learn`
- `numpy==1.26.4`

---

## ⚙️ Configuration

- **Gemini API Key** is required for image analysis and recipe generation.  
  Set via Streamlit secrets as shown in the installation section.
- **Caching** is used for API status checks and model initialization for efficiency.

---

## 🧪 Examples

- Upload a photo of ingredients like: 🥦 broccoli, 🍗 chicken, 🍚 rice.
- Choose **Asian** cuisine and **Medium** difficulty.
- Get back a full recipe with steps, nutrition stats, and health score.
- View similar recipes from your history with AI recommendations.

---

## 🛠️ Troubleshooting

- **Gemini not configured?**  
  Ensure your API key is correct and placed in `.streamlit/secrets.toml`.

- **Ingredient detection failed?**  
  Use manual entry for more precise control.

- **App won't start?**  
  Confirm all dependencies are installed with `pip install -r requirements.txt`.

---

## 👥 Contributors

- [Your Name] – Developer & Maintainer  
(Replace this section with actual contributors.)

---

## 📄 License

MIT License. See `LICENSE` file for details.
