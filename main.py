from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from openai import OpenAI
from dotenv import load_dotenv
import io
import json

load_dotenv()

# =========================
# AI CHAT (OpenAI)
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Load TFLite model (ONCE AT STARTUP)
# =========================
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Load labels
with open("labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

app = FastAPI(title="DermAware Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Utils
# =========================
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224), Image.BILINEAR)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# =========================
# IMAGE CLASSIFICATION
# =========================
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_data = preprocess_image(image)

        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])[0]

        max_idx = int(np.argmax(output_data))
        label = labels[max_idx]
        confidence = float(output_data[max_idx])

        return JSONResponse({
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image classification failed: {str(e)}"
        )

# =========================
# AI CHAT
# =========================
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dermatology assistant for a mobile application. "
                        "Provide general, safe skin-care advice only. "
                        "Do NOT give medical diagnosis."
                    )
                },
                {
                    "role": "user",
                    "content": req.message
                }
            ]
        )

        return {
            "reply": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI chat failed: {str(e)}"
        )

# =========================
# AI RESULT EXPLANATION (STRUCTURED)
# =========================
class ExplainRequest(BaseModel):
    label: str

@app.post("/explain_result")
def explain_result(req: ExplainRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dermatology assistant. "
                        "Given a skin condition name, provide information in JSON format with these fields:\n"
                        "- explanation: A concise description of the condition (2-3 sentences)\n"
                        "- causes: Array of 3-5 common causes\n"
                        "- dos: Array of 3-5 recommended actions\n"
                        "- donts: Array of 3-5 things to avoid\n\n"
                        "Return ONLY valid JSON, no markdown formatting."
                    )
                },
                {
                    "role": "user",
                    "content": f"Explain the skin condition: {req.label}"
                }
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        
        # Parse JSON response
        try:
            data = json.loads(content)
            return {
                "explanation": data.get("explanation", "No explanation available"),
                "causes": data.get("causes", []),
                "dos": data.get("dos", []),
                "donts": data.get("donts", [])
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "explanation": content,
                "causes": [],
                "dos": [],
                "donts": []
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI explanation failed: {str(e)}"
        )

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": True}