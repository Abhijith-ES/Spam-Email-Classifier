# --- ML Training and Model Persistence Script ---
# 1. Loads configuration from settings.yaml.
# 2. Loads the spam dataset.
# 3. Trains four classification models (LR, NB, SVM, RF).
# 4. Evaluates all models and selects the "MODEL_TO_USE" defined in the config.
# 5. Saves the models dictionary and the fitted TF-IDF vectorizer using pickle.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import yaml
import os

# --- 1. Load Configuration ---
try:
    with open('config/settings.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    print("✅ Configuration loaded from config/settings.yaml.")
except FileNotFoundError:
    print("FATAL ERROR: config/settings.yaml not found.")
    exit()

# Get settings from config
DATA_FILE = CONFIG['DATA_FILE']
ENCODING = CONFIG['ENCODING']
TEST_SIZE = CONFIG['TEST_SIZE']
RANDOM_STATE = CONFIG['RANDOM_STATE']
MODEL_TO_USE = CONFIG['MODEL_TO_USE']

# --- 2. Load and Prepare Data ---
try:
    df = pd.read_csv(DATA_FILE, encoding=ENCODING)
    # Assuming the input CSV structure used in the original prompt (first two columns)
    df.columns = ["label", "message"]
    print(f"✅ Data loaded successfully from {DATA_FILE}. Total rows: {len(df)}")
except Exception as e:
    print(f"FATAL ERROR: Could not load data file {DATA_FILE}. Error: {e}")
    exit()

# Map labels to numeric values (ham: 0, spam: 1)
X = df['message']
y = df['label'].map({'ham': 0, 'spam': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# --- 3. Feature Extraction (TF-IDF) ---
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("✅ TF-IDF Vectorizer fitted and data transformed.")


# --- 4. Define and Train Models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(kernel='linear', random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE)
}

print("\n--- Model Training & Evaluation ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    try:
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        print(f"Model: {name}")
        print(classification_report(y_test, preds))
        # print(confusion_matrix(y_test, preds))
    except Exception as e:
        print(f"Error training {name}: {e}")

# --- 5. Save Artifacts for Prediction ---

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Save the final selected model
try:
    final_model = models[MODEL_TO_USE]
    with open('models/final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)

    # Save the fitted vectorizer
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"\n======================================")
    print(f"🎉 Success! The model '{MODEL_TO_USE}' and vectorizer have been saved to the 'models/' directory.")
    print(f"======================================")

except KeyError:
    print(f"\nFATAL ERROR: The model name '{MODEL_TO_USE}' in settings.yaml does not match any trained models.")
    print("Available models: " + ", ".join(models.keys()))
except Exception as e:
    print(f"\nFATAL ERROR: Failed to save model artifacts: {e}")