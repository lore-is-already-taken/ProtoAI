import os
from datetime import datetime

from dotenv import load_dotenv
from firebase_admin import credentials, firestore, initialize_app

load_dotenv()

# Load Firebase credentials from the environment variable
cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
cred = credentials.Certificate(cred_path)
initialize_app(cred)
db = firestore.client()


def save_to_firestore(data, collection="responses"):
    """Saves data to Firestore in the specified collection with tags and a custom doc ID."""
    try:
        detected_object = (
            data.get("image_analysis", {}).get("detected_object", "").strip().lower()
        )
        if not detected_object or detected_object == "unknown":
            detected_object = "unlabeled"

        existing_tags = data.get("tags", [])
        tags = set(existing_tags + [detected_object])  # evitar duplicados
        data["tags"] = list(tags)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        doc_id = f"{detected_object}_{timestamp}"

        doc_ref = db.collection(collection).document(doc_id)
        doc_ref.set(data)

        print(f"✅ Data saved with doc ID: {doc_id} in collection: {collection}")
    except Exception as e:
        print(f"❌ Error saving data to Firestore: {e}")
