import os
import scipy.io
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np

cred = credentials.Certificate("credentials/credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

folder_path = "./ninaproDB1"

for filename in os.listdir(folder_path):
    if filename.endswith(".mat"):
        mat_path = os.path.join(folder_path, filename)

        try:
            data = scipy.io.loadmat(mat_path)

            if "emg" not in data:
                print(f"⚠️ El archivo {filename} no contiene 'emg'")
                continue

            emg_array = data['emg']  
            subject = filename.split("_")[0]
            exercise = filename.split("_")[2].split(".")[0]

            truncated = emg_array[:100]
            emg_by_channel = {f"ch{i}": truncated[:, i].tolist() for i in range(truncated.shape[1])}

            doc_data = {
                "subject": subject,
                "exercise": exercise,
                "shape": list(emg_array.shape),
                "emg_by_channel": emg_by_channel
            }

            doc_id = filename.replace(".mat", "")
            db.collection("emg_signals").document(doc_id).set(doc_data)
            print(f"✅ Subido: {doc_id}")

        except Exception as e:
            print(f"❌ Error con {filename}: {e}")
