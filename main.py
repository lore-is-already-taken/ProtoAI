import os
import time  

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from firebase import save_to_firestore
from models import ImageRequest
from openAI.client import OpenAIClient
from prompts.prompts import HAND_PROMPT_3

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "gpt-4o" # Default model
API_KEY = os.getenv("OPENAI_API_KEY")
RASP_API_URL = os.getenv("RASP_API_URL")


@app.get("/")
def ping():
    return {"response": "OK"}


@app.websocket("/ws")
async def websocket_server(websocket: WebSocket):
    """WebSocket server to receive real-time keypoints."""
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
    if request.data:
        selected_model = MODEL
        client = OpenAIClient(api_key=API_KEY)
        image_data = request.data

        start_time = time.time()

        # 1️⃣ Detección de objeto
        detected_object_response = client.detect_object(selected_model, image_data)
        detected_object = client.extract_object_from_response(detected_object_response)

        # 2️⃣ Movimiento sugerido
        suggested_movement_response = client.generate_suggested_movement(
            selected_model, detected_object
        )
        suggested_movement = client.extract_suggested_movement(suggested_movement_response)

        # 3️⃣ Movimientos de mano
        hand_movements_response = client.generate_hand_movements(
            HAND_PROMPT_3, selected_model, image_data
        )

        cleaned_hand_movements = hand_movements_response.strip()
        if not cleaned_hand_movements.startswith("["):
            cleaned_hand_movements = f"[{cleaned_hand_movements}]"

        # ⏱️ Tiempo total
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        # 4️⃣ Guardar en Firestore
        tags = [detected_object.lower(), selected_model]
        final_response = {
            "hand_movements": cleaned_hand_movements,
            "image_analysis": {
                "detected_object": detected_object,
                "suggested_movement": suggested_movement,
            },
            "provided_image": image_data,
            "llm_model": selected_model,
            "tags": tags,
            "processing_time_seconds": processing_time,
        }

        save_to_firestore(final_response)
        return final_response
