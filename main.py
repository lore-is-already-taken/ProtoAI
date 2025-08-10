import os
import time
from typing import Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from firebase import db, save_to_firestore
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

MODEL = "gpt-4.1-nano"
API_KEY = os.getenv("OPENAI_API_KEY")
RASP_API_URL = os.getenv("RASP_API_URL")


def get_emg_context(label_detected: str) -> Optional[Tuple[str, str]]:
    tag_to_exercise = {
        "bottle": ["E2", "EA1"],
        "cup": ["E3", "EA2"],
        "phone": ["E5", "EA5"],
    }

    label = label_detected.lower()
    matched_exercises = next(
        (e for k, e in tag_to_exercise.items() if k in label), None
    )

    if not matched_exercises:
        print(f"⚠️ No EMG mapping for label: '{label_detected}'")
        return None

    context_lines = []
    used_sources = set()

    for collection_name in ["emg_signals", "emg_signals_db2"]:
        docs = db.collection(collection_name).stream()
        for doc in docs:
            emg_data = doc.to_dict()
            exercise_val = emg_data.get("exercise", "")
            if exercise_val not in matched_exercises:
                continue

            subject = emg_data.get("subject", "unknown")
            shape = emg_data.get("shape", [])
            emg_channels = emg_data.get("emg_by_channel", {})

            # Construir un resumen de todos los canales disponibles
            all_channels = [
                f"ch{i}: {emg_channels.get(f'ch{i}', [])[:3]}"
                for i in range(len(emg_channels))
            ]
            context_lines.append(
                f"- {collection_name.upper()} / Subject {subject}, shape {shape}, "
                + " | ".join(all_channels)
            )
            used_sources.add(collection_name.upper())

    if not context_lines:
        print(f"⚠️ No EMG document found for exercises: {matched_exercises}")
        return None

    emg_context = (
        f"Exercises {', '.join(matched_exercises)} matched to the detected object.\n"
        "EMG signals were retrieved from the following sources:\n"
        + "\n".join(context_lines)
        + "\nUse this signal information to refine the hand movement instruction."
    )

    return emg_context, ", ".join(sorted(used_sources))


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
    emg_result_tuple = get_emg_context(detected_object)
    if emg_result_tuple:
        emg_context, emg_source = emg_result_tuple
        print(f"✅ EMG sources used: {emg_source}")
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
                "emg_source": emg_source,
            },
            "provided_image": image_data,
            "llm_model": selected_model,
            "tags": [detected_object.lower(), selected_model, emg_source, "fusion_emg"],
            "processing_time_seconds": round(time.time() - start_time, 2),
        }
        save_to_firestore(emg_result, collection="responses_emg_llm")
    else:
        print("🚫 No EMG fusion generated.")

    return {"status": "OK", "detected_object": detected_object}
