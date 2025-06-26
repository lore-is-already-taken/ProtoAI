import os

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

MODEL = "gpt-4o"
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
            print(f"Received keypoints: {data}")  # JSON string
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await websocket.close()


@app.post("/upload-image")
def upload_image(request: ImageRequest):
    if request.data:
        # Usar modelo enviado desde el frontend, o por defecto usar el predefinido
        selected_model = request.model or MODEL

        client = OpenAIClient(api_key=API_KEY)
        image_data = request.data

        # 1️⃣ Detect the object
        detected_object_response = client.detect_object(selected_model, image_data)
        detected_object = client.extract_object_from_response(detected_object_response)

        # 2️⃣ Ask OpenAI what movement to perform
        suggested_movement_response = client.generate_suggested_movement(
            selected_model, detected_object
        )
        suggested_movement = client.extract_suggested_movement(
            suggested_movement_response
        )

        # 3️⃣ Generate hand movements based on the object
        hand_movements_response = client.generate_hand_movements(
            HAND_PROMPT_3, selected_model, image_data
        )
        cleaned_hand_movements = (
            hand_movements_response.replace("```json", "").replace("```", "").strip()
        )

        # 4️⃣ Save the final response in Firebase
        final_response = {
            "hand_movements": cleaned_hand_movements,
            "image_analysis": {
                "detected_object": detected_object,
                "suggested_movement": suggested_movement,
            },
            "provided_image": image_data,
            "llm_model": selected_model,  # Model used for LLM
        }

        save_to_firestore(final_response)
        return final_response
