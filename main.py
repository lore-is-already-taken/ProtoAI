import os
import time
from typing import Optional
import statistics

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from firebase import save_to_firestore, db
from models import ImageRequest
from openAI.client import OpenAIClient
from prompts.prompts import HAND_PROMPT_3, HAND_PROMPT_EMG_FUSION

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "gpt-4o-mini"
API_KEY = os.getenv("OPENAI_API_KEY")
RASP_API_URL = os.getenv("RASP_API_URL")


def get_emg_context(label_detected: str) -> Optional[str]:
    tag_to_exercise = {
        "bottle": "E2",
        "cup": "E3",
        "phone": "E5",
        # Agrega más si es necesario
    }

    label = label_detected.lower()
    matched_exercise = None

    for keyword, exercise in tag_to_exercise.items():
        if keyword in label:
            matched_exercise = exercise
            print(f"✅ EMG match: '{keyword}' in '{label_detected}' → {exercise}")
            break

    if not matched_exercise:
        print(f"⚠️ No EMG mapping for label: '{label_detected}'")
        return None

    # Recopilar información de todos los sujetos para ese ejercicio
    context_lines = []
    docs = db.collection("emg_signals").stream()
    for doc in docs:
        emg_data = doc.to_dict()
        if emg_data.get("exercise") == matched_exercise:
            subject = emg_data.get("subject", "unknown")
            shape = emg_data.get("shape", [])
            ch0 = emg_data.get("emg_by_channel", {}).get("ch0", [])[:5]
            context_lines.append(
                f"- Subject {subject}, shape {shape}, ch0 sample: {ch0}"
            )

    if not context_lines:
        print(f"⚠️ No EMG document found for exercise: '{matched_exercise}'")
        return None

    emg_context = (
        f"Exercise {matched_exercise} from NinaPro DB1 was matched to the detected object.\n"
        "Multiple subjects recorded EMG signals as follows:\n"
        + "\n".join(context_lines)
        + "\nUse this signal information to refine the hand movement instruction."
    )

    return emg_context


@app.get("/")
def ping():
    return {"response": "OK"}


@app.websocket("/ws")
async def websocket_server(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received keypoints: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await websocket.close()


@app.post("/upload-image")
def upload_image(request: ImageRequest):
    if not request.data:
        return {"error": "No image data provided."}

    selected_model = MODEL
    client = OpenAIClient(api_key=API_KEY)
    image_data = request.data
    start_time = time.time()

    # 1️⃣ Detección de objeto
    detected_object_response = client.detect_object(selected_model, image_data)
    detected_object = client.extract_object_from_response(detected_object_response)
    print(f"🧠 Detected object: '{detected_object}'")

    # 2️⃣ Movimiento sugerido
    suggested_movement_response = client.generate_suggested_movement(
        selected_model, detected_object
    )
    suggested_movement = client.extract_suggested_movement(suggested_movement_response)

    # 3️⃣ Instrucciones estándar (solo visión)
    vision_hand_response = client.generate_hand_movements(
        HAND_PROMPT_3, selected_model, image_data
    )
    cleaned_vision_response = vision_hand_response.strip()
    if not cleaned_vision_response.startswith("["):
        cleaned_vision_response = f"[{cleaned_vision_response}]"

    # 4️⃣ Guardar respuesta solo visión
    vision_result = {
        "hand_movements": cleaned_vision_response,
        "image_analysis": {
            "detected_object": detected_object,
            "suggested_movement": suggested_movement,
        },
        "provided_image": image_data,
        "llm_model": selected_model,
        "tags": [detected_object.lower(), selected_model],
        "processing_time_seconds": round(time.time() - start_time, 2),
    }
    save_to_firestore(vision_result, collection="responses")

    # 5️⃣ Intentar generar respuesta fusionada con EMG
    emg_context = get_emg_context(detected_object)
    if emg_context:
        print("🔬 EMG context retrieved. Generating fused response...")
        fusion_prompt = HAND_PROMPT_EMG_FUSION.replace("{EMG_CONTEXT}", emg_context)
        emg_response = client.generate_hand_movements(
            fusion_prompt, selected_model, image_data
        )
        cleaned_emg_response = emg_response.strip()
        if not cleaned_emg_response.startswith("["):
            cleaned_emg_response = f"[{cleaned_emg_response}]"

        emg_result = {
            "hand_movements": cleaned_emg_response,
            "image_analysis": {
                "detected_object": detected_object,
                "suggested_movement": suggested_movement,
                "emg_context": emg_context,
            },
            "provided_image": image_data,
            "llm_model": selected_model,
            "tags": [detected_object.lower(), selected_model, "fusion_emg"],
            "processing_time_seconds": round(time.time() - start_time, 2),
        }
        save_to_firestore(emg_result, collection="responses_emg_llm")
    else:
        print("🚫 No EMG fusion generated.")

    return {"status": "OK", "detected_object": detected_object}
