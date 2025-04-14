import json
import re

import openai

from prompts.prompts import MOVEMENT_ANALYSIS_PROMPT, OBJECT_DETECTION_PROMPT
from utils.images import encode_image


class OpenAIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def detect_object(self, model, image_url):
        """Asks OpenAI to identify the main object in the image."""
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": OBJECT_DETECTION_PROMPT},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=50,
            )

            vision_response = response.choices[0].message.content.strip()
            print("\n🔍 RAW OpenAI Response (Object Detection):\n", vision_response)

            return vision_response

        except Exception as e:
            print(f"❌ Error detecting object: {e}")
            return "Error detecting object"

    def generate_suggested_movement(self, model, detected_object):
        """Asks OpenAI what the prosthetic hand should do with the detected object."""
        movement_prompt = MOVEMENT_ANALYSIS_PROMPT.format(
            detected_object=detected_object
        )

        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": movement_prompt}],
                max_tokens=50,
            )

            vision_response = response.choices[0].message.content.strip()
            print("\n🖐️ RAW OpenAI Response (Suggested Movement):\n", vision_response)

            return vision_response

        except Exception as e:
            print(f"❌ Error generating suggested movement: {e}")
            return "Error generating movement"

    def generate_hand_movements(self, prompt, model, image_url):
        """Asks OpenAI to generate hand movement commands based on the image."""
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=300,
            )

            vision_response = response.choices[0].message.content.strip()
            print("\n🤖 RAW OpenAI Response (Hand Movements):\n", vision_response)

            return vision_response

        except Exception as e:
            print(f"❌ Error generating hand movements: {e}")
            return "Error generating hand movements"

    def extract_object_from_response(self, response_text):
        """Extracts the detected object name from OpenAI's response."""
        match = re.search(r"Detected object:\s*([\w\s]+)", response_text, re.IGNORECASE)
        return match.group(1).strip() if match else "unknown"

    def extract_suggested_movement(self, response_text):
        """Extracts the suggested movement from OpenAI's response."""
        match = re.search(
            r"Suggested movement:\s*([\w\s]+)", response_text, re.IGNORECASE
        )
        return match.group(1).strip() if match else "none"
