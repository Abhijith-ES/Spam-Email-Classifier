# --- Gmail Real-Time Spam Classifier ---
# This script loads a trained model and uses the Google Gmail API to fetch
# unread emails, classify them as SPAM or NOT SPAM, and summarize the results.

import os
import pickle
import base64
import yaml
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import pandas as pd
import httplib2

# --- 1. Load Configuration ---
try:
    with open('config/settings.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    print("✅ Configuration loaded from config/settings.yaml.")
except FileNotFoundError:
    print("FATAL ERROR: config/settings.yaml not found. Check your project structure.")
    exit()

# --- 2. Configuration Variables ---
MODEL_TO_USE = CONFIG['MODEL_TO_USE']
QUERY = CONFIG['QUERY']
MAX_EMAILS = CONFIG['MAX_EMAILS']
TOKEN_FILE = CONFIG['TOKEN_FILE']
CREDENTIALS_FILE = CONFIG['CREDENTIALS_FILE']
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# --- 3. Load Model Artifacts ---
# These files are created by src/01_train_and_save.py
try:
    with open('models/final_model.pkl', 'rb') as f:
        models = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"✅ Loaded {MODEL_TO_USE} model and TF-IDF vectorizer.")
except FileNotFoundError:
    print("FATAL ERROR: Models not found. Run src/01_train_and_save.py first!")
    exit()

# --- 4. Gmail Authentication (OAuth 2.0 Installed Application Flow) ---

def authenticate_gmail():
    """Handles the OAuth 2.0 flow for a desktop application."""
    creds = None
    # Check if we already have a token file (token.json)
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Refresh the token if it's expired
            print("Token expired, refreshing...")
            creds.refresh(httplib2.Http())
        else:
            # Start the full flow if no token exists
            print(f"Opening browser for Gmail authentication (using {CREDENTIALS_FILE})...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
            print(f"✅ Credentials saved to {TOKEN_FILE}.")

    return build('gmail', 'v1', credentials=creds)

# --- 5. Helper Functions for Email Parsing ---

def get_text_from_part(part):
    """Decodes Base64 data and extracts text from a message part."""
    data = part.get('body', {}).get('data')
    if data:
        try:
            text = base64.urlsafe_b64decode(data.encode('ASCII')).decode('utf-8', errors='ignore')
            if part.get('mimeType') == 'text/html':
                # Use BeautifulSoup to strip HTML tags, leaving only readable text
                text = BeautifulSoup(text, 'html.parser').get_text(separator=' ', strip=True)
            return text
        except (base64.binascii.Error, UnicodeDecodeError) as e:
            # Handle decoding errors gracefully
            print(f"Decoding error: {e}")
            return None
    return None

def extract_email_text(msg):
    """Recursively attempts to extract the plain text body from the email message."""
    payload = msg.get('payload', {})
    headers = payload.get('headers', [])

    # Extract Subject and From headers
    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), "(No Subject)")
    sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), "(No Sender)")

    # Try to get text from the main payload or snippet
    text = get_text_from_part(payload)
    
    # If not found, iterate through parts (for multipart messages)
    if not text and 'parts' in payload:
        for p in payload['parts']:
            text = get_text_from_part(p)
            if text:
                break
    
    # Fallback to the message snippet if body extraction fails
    if not text:
        text = msg.get('snippet', '')
        
    # Clean up whitespace
    text = " ".join(text.split())
    return subject, sender, text

# --- 6. Main Classification Logic ---

def classify_emails(service):
    """Fetches emails and uses the trained model to classify them."""
    print(f"\n🔍 Searching for emails using query: '{QUERY}' (Max: {MAX_EMAILS})...")
    
    try:
        # Fetch list of messages matching the query
        resp = service.users().messages().list(userId='me', q=QUERY, maxResults=MAX_EMAILS).execute()
    except Exception as e:
        print(f"API Error during list operation: {e}")
        return

    msgs = resp.get('messages', [])
    
    if not msgs:
        print("🎉 No messages found matching the query. Exiting.")
        return

    print(f"📧 Found {len(msgs)} messages. Starting classification with {MODEL_TO_USE}...")

    results = []
    
    for i, mmeta in enumerate(msgs, start=1):
        try:
            # Get the full message content
            msg = service.users().messages().get(userId='me', id=mmeta['id'], format='full').execute()
            subject, sender, body = extract_email_text(msg)
            
            # Combine subject and body for classification
            full_text = subject + " " + body
            
            # Vectorize the text using the saved TF-IDF vectorizer
            X_tfidf = vectorizer.transform([full_text])
            
            # Get the prediction from the selected model
            model_instance = models[MODEL_TO_USE]
            pred = int(model_instance.predict(X_tfidf)[0])
            
            label = "SPAM" if pred == 1 else "NOT SPAM"
            
            print(f"\n[{i}/{len(msgs)}] From: {sender}")
            print(f"Subject: {subject[:70]}...")
            print(f"Prediction: {label}")
            
            results.append((sender, subject, label))

        except Exception as e:
            print(f"\n[Error processing email {i}]: {e}")
            results.append(("(Error)", "(Error)", "CLASSIFICATION FAILED"))
            
    # --- 7. Summary Table ---
    df = pd.DataFrame(results, columns=["Sender", "Subject", "Prediction"])
    print("\n\n========== CLASSIFICATION SUMMARY ==========")
    print(df.to_string())

if __name__ == '__main__':
    service = authenticate_gmail()
    classify_emails(service)