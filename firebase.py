import os

from dotenv import load_dotenv
from firebase_admin import credentials, firestore, initialize_app

load_dotenv()

# Load Firebase credentials from the environment variable
cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
cred = credentials.Certificate(cred_path)
initialize_app(cred)
db = firestore.client()


def save_to_firestore(data):
    """Saves data to Firestore with tags and custom doc ID based on detected_object."""
    try:
        # Obtener objeto detectado y sanitizarlo
        detected_object = data.get("image_analysis", {}).get("detected_object", "").strip().lower()
        if not detected_object or detected_object == "unknown":
            detected_object = "unlabeled"

        # Crear campo de etiquetas (solo el objeto detectado por ahora)
        data["tags"] = [detected_object]

        # Crear un ID de documento legible
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        doc_id = f"{detected_object}_{timestamp}"

        # Guardar en Firestore
        doc_ref = db.collection("responses").document(doc_id)
        doc_ref.set(data)

        print(f"✅ Data saved with doc ID: {doc_id} and tag: {detected_object}")
    except Exception as e:
        print(f"❌ Error saving data to Firestore: {e}")
