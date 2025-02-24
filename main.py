import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, firestore, initialize_app
from pydantic import BaseModel

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

IMAGE_PROMPT = HAND_PROMPT_3
MODEL = "gpt-4o-mini"

API_KEY = os.getenv("OPENAI_API_KEY")
RASP_API_URL = str(os.getenv("RASP_API_URL"))


@app.websocket("/ws")
async def websocket_server(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received keypoints: {data}")  # JSON string
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await websocket.close()


cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
cred = credentials.Certificate(cred_path)
initialize_app(cred)
db = firestore.client()


@app.get("/")
def ping():
    return {"response": "OK"}


class image(BaseModel):
    data: str


def save_to_firestore(data):
    try:
        doc_ref = db.collection("responses").document()
        doc_ref.set(data)
        print("Data saved successfully in Firestore")
    except Exception as e:
        print(f"Error saving data to Firestore: {e}")


@app.post("/upload-image")
def upload_image(request: image):
    if request.data:
        client = OpenAIClient(api_key=API_KEY)
        prompt = IMAGE_PROMPT
        image_data = request.data
        response = client.process_image(prompt, MODEL, image_data)
        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        print(cleaned_response)
        save_to_firestore({"response": cleaned_response})

        return {"response": cleaned_response}


def post_data(instructions):
    url = RASP_API_URL
    headers = {"Content-Type": "application/json"}

    payload = {"data": instructions}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print("Data sent successfully")
        print(response.json())
    else:
        print("Error sending data")
        print(response.json())
