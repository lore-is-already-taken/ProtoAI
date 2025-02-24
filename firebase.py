import os
from firebase_admin import firestore, credentials, initialize_app
from dotenv import load_dotenv

load_dotenv()

# Load Firebase credentials from the environment variable
cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
cred = credentials.Certificate(cred_path)
initialize_app(cred)
db = firestore.client()

def save_to_firestore(data):
    """Saves data to Firestore."""
    try:
        doc_ref = db.collection("responses").document()
        doc_ref.set(data)
        print("Data saved successfully in Firestore")
    except Exception as e:
        print(f"Error saving data to Firestore: {e}")
