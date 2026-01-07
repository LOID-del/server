from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
import io
import json
import httpx
import traceback

load_dotenv()

# =========================
# AI CHAT (OpenAI)
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# AILABTOOLS CONFIG
# =========================
AILABTOOLS_API_KEY = os.getenv("AILABTOOLS_API_KEY")

# =========================
# TFLite Model (OPTIONAL - only if files exist)
# =========================
interpreter = None
labels = []

try:
    import tensorflow as tf
    if os.path.exists("model_unquant.tflite") and os.path.exists("labels.txt"):
        interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
        interpreter.allocate_tensors()
        with open("labels.txt") as f:
            labels = [line.strip() for line in f.readlines()]
        print("✅ TFLite model loaded")
    else:
        print("⚠️ TFLite files not found - offline mode disabled")
except Exception as e:
    print(f"⚠️ TFLite unavailable: {e}")

app = FastAPI(title="DermAware Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (optional)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# Utils
# =========================
def preprocess_image(image: Image.Image):
    """Preprocess image for TFLite model"""
    image = image.resize((224, 224), Image.BILINEAR)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def normalize_label(label: str) -> str:
    """Normalize condition labels"""
    label = label.lower().strip()
    
    if any(term in label for term in ["healthy", "normal", "clear"]):
        return "healthy"
    
    if any(term in label for term in ["not skin", "not_skin", "no skin", "invalid"]):
        return "not_skin"
    
    return label

def is_low_confidence(confidence: float, threshold: float = 0.4) -> bool:
    """Check if confidence is too low"""
    return confidence < threshold

def get_skin_info_from_openai(label: str):
    """Get skin condition info from OpenAI"""
    try:
        print(f"🤖 Asking OpenAI for details about: {label}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dermatology assistant. "
                        "Provide info in JSON format:\n"
                        "- alsoKnownAs: The most popular common name for this condition\n"
                        "- explanation: 2-3 sentences definition in English\n"
                        "- causes: 3-5 common causes in English\n"
                        "- dos: 3-5 care recommendations in English\n"
                        "- donts: 3-5 things to avoid in English\n"
                        "Return ONLY valid JSON."
                    )
                },
                {"role": "user", "content": f"Explain this skin condition: {label}"}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        return {
            "alsoKnownAs": data.get("alsoKnownAs", label),
            "explanation": data.get("explanation", "Information currently unavailable."),
            "causes": data.get("causes", []),
            "dos": data.get("dos", []),
            "donts": data.get("donts", [])
        }
    except Exception as e:
        print(f"❌ Error getting OpenAI info: {e}")
        return {
            "alsoKnownAs": label,
            "explanation": "Information currently unavailable.",
            "causes": [],
            "dos": [],
            "donts": []
        }

# =========================
# ONLINE: AILABTOOLS API + OpenAI Info
# =========================
@app.post("/classify/online")
async def classify_online(file: UploadFile = File(...)):
    """Online classification with OpenAI info for skin conditions"""
    try:
        print(f"📥 Received file: {file.filename}, type: {file.content_type}")
        
        contents = await file.read()
        print(f"📦 File size: {len(contents)} bytes")

        if len(contents) == 0:
            raise HTTPException(400, "Empty file received")

        try:
            img = Image.open(io.BytesIO(contents))
            print(f"✅ Valid image: {img.format} {img.size}")
        except Exception as e:
            raise HTTPException(400, f"Invalid image file: {str(e)}")

        # Call Ailabtools API
        url = "https://www.ailabapi.com/api/portrait/analysis/skin-disease-detection"
        print(f"🌐 Calling Ailabtools API...")

        headers = {
            "ailabapi-api-key": AILABTOOLS_API_KEY
        }

        files = {
            "image": ("photo.jpg", contents, "image/jpeg")
        }

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.post(
                url,
                files=files,
                headers=headers
            )

        print(f"📡 Ailabtools status: {response.status_code}")
        
        if response.status_code != 200:
            error_text = response.text[:500]
            print(f"❌ Ailabtools error: {error_text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Ailabtools API error: {error_text}"
            )

        result = response.json()
        print(f"📊 Ailabtools response: {json.dumps(result, indent=2)}")

        error_code = result.get("error_code", 0)
        if error_code != 0:
            error_msg = result.get("error_msg", "Unknown error")
            print(f"⚠️ Ailabtools error_code: {error_code}, msg: {error_msg}")
            
            return JSONResponse({
                "label": "Not Skin",
                "confidence": 0.0,
                "error": "api_error",
                "error_message": error_msg
            })

        data = result.get("data", {})
        results = data.get("results_english", {})
        
        if not results:
            print("⚠️ No results from Ailabtools")
            return JSONResponse({
                "label": "Unknown",
                "confidence": 0.0,
                "error": "no_results"
            })

        print(f"✅ Classification results: {results}")

        best_label = max(results.items(), key=lambda x: x[1])
        label = best_label[0].replace("_", " ").title()
        confidence = float(best_label[1])

        print(f"🏆 Best: {label} = {confidence:.2%}")

        normalized = normalize_label(label)

        if normalized not in ["healthy", "not_skin"]:
            print(f"🔍 Detected skin condition: {label}, getting OpenAI info...")
            openai_info = get_skin_info_from_openai(label)
            
            return JSONResponse({
                "label": label,
                "confidence": confidence,
                "all_results": results,
                "alsoKnownAs": openai_info["alsoKnownAs"],
                "explanation": openai_info["explanation"],
                "causes": openai_info["causes"],
                "dos": openai_info["dos"],
                "donts": openai_info["donts"]
            })
        else:
            print(f"✅ Result is {normalized}, no OpenAI info needed")
            return JSONResponse({
                "label": label,
                "confidence": confidence,
                "all_results": results
            })

    except httpx.TimeoutException as e:
        print(f"⏱️ Timeout: {e}")
        raise HTTPException(504, "Request to Ailabtools timed out")
    
    except httpx.RequestError as e:
        print(f"🌐 Network error: {e}")
        raise HTTPException(503, f"Network error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"📋 Traceback:\n{traceback.format_exc()}")
        raise HTTPException(500, f"Classification failed: {str(e)}")

# =========================
# OFFLINE: LOCAL TFLITE MODEL
# =========================
@app.post("/classify/offline")
async def classify_offline(file: UploadFile = File(...)):
    """Offline analysis using local TFLite model"""
    
    if not interpreter:
        raise HTTPException(503, "Offline mode not available on this server")
    
    try:
        print(f"📥 Offline: {file.filename}")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_data = preprocess_image(image)

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])[0]

        max_idx = int(np.argmax(output_data))
        raw_label = labels[max_idx]
        confidence = float(output_data[max_idx])

        print(f"🔍 TFLite: {raw_label} ({confidence:.2%})")

        normalized = normalize_label(raw_label)

        if normalized == "not_skin":
            return JSONResponse({
                "label": "Not Skin",
                "confidence": confidence,
                "error": "not_skin"
            })

        if is_low_confidence(confidence, 0.35):
            return JSONResponse({
                "label": "Not Skin",
                "confidence": confidence,
                "error": "low_confidence"
            })

        if normalized == "healthy":
            return JSONResponse({
                "label": "Healthy Skin",
                "confidence": confidence
            })

        return JSONResponse({
            "label": raw_label,
            "confidence": confidence
        })

    except Exception as e:
        print(f"❌ Offline error: {e}")
        print(traceback.format_exc())
        raise HTTPException(500, f"Offline failed: {str(e)}")

# =========================
# UNIFIED ENDPOINT
# =========================
@app.post("/classify")
async def classify_unified(
    file: UploadFile = File(...),
    mode: str = Form("online")
):
    """Unified endpoint that routes to online or offline"""
    print(f"📍 Mode: {mode}")
    
    if mode.lower() == "online":
        return await classify_online(file)
    else:
        return await classify_offline(file)

# =========================
# AI CHAT
# =========================
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        print(f"💬 Chat: {req.message[:50]}...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dermatology assistant for a mobile app. "
                        "Provide general skincare advice. "
                        "Do NOT diagnose. "
                        "Always recommend consulting healthcare professionals."
                    )
                },
                {"role": "user", "content": req.message}
            ]
        )

        return {"reply": response.choices[0].message.content}

    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(500, f"Chat failed: {str(e)}")

# =========================
# AI EXPLANATION
# =========================
class ExplainRequest(BaseModel):
    label: str

@app.post("/explain_result")
def explain_result(req: ExplainRequest):
    try:
        print(f"📖 Explain: {req.label}")
        
        condition = req.label.lower().strip()
        
        if "not skin" in condition:
            return {
                "explanation": "The image doesn't appear to contain skin. Please take a clear photo of skin.",
                "causes": [],
                "dos": [
                    "Ensure good lighting",
                    "Focus on skin area",
                    "Keep camera steady",
                    "Take from 6-12 inches away"
                ],
                "donts": [
                    "Don't take photos of non-skin",
                    "Avoid blurry images",
                    "Don't include too much background",
                    "Avoid extreme close-ups"
                ]
            }
        
        if "healthy" in condition:
            return {
                "explanation": "Your skin appears healthy! Continue good skincare habits.",
                "causes": [],
                "dos": [
                    "Maintain skincare routine",
                    "Stay hydrated",
                    "Use SPF 30+ daily",
                    "Get 7-9 hours sleep",
                    "Exercise regularly"
                ],
                "donts": [
                    "Don't skip sunscreen",
                    "Don't over-wash (max 2x daily)",
                    "Don't pick at skin",
                    "Avoid harsh products",
                    "Don't skip moisturizer"
                ]
            }
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dermatology assistant. "
                        "Provide info in JSON format:\n"
                        "- explanation: 2-3 sentences with disclaimer to see doctor\n"
                        "- causes: 3-5 common causes\n"
                        "- dos: 3-5 care recommendations\n"
                        "- donts: 3-5 things to avoid\n"
                        "Return ONLY valid JSON."
                    )
                },
                {"role": "user", "content": f"Explain: {req.label}"}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        print(f"✅ Got AI explanation")
        
        try:
            data = json.loads(content)
            return {
                "explanation": data.get("explanation", "No info available"),
                "causes": data.get("causes", []),
                "dos": data.get("dos", []),
                "donts": data.get("donts", [])
            }
        except json.JSONDecodeError:
            return {
                "explanation": content,
                "causes": [],
                "dos": [],
                "donts": []
            }

    except Exception as e:
        print(f"❌ Explain error: {e}")
        raise HTTPException(500, f"Explanation failed: {str(e)}")

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "DermAware Backend",
        "version": "2.0",
        "environment": "production" if os.getenv("RAILWAY_ENVIRONMENT") else "development",
        "ailabtools": "✅" if AILABTOOLS_API_KEY else "❌",
        "openai": "✅" if os.getenv("OPENAI_API_KEY") else "❌",
        "tflite": "✅" if interpreter else "❌ (cloud mode only)",
        "endpoints": {
            "health": "GET /",
            "online": "POST /classify/online",
            "offline": "POST /classify/offline (requires TFLite)",
            "unified": "POST /classify",
            "chat": "POST /chat",
            "explain": "POST /explain_result"
        }
    }

# =========================
# UI PAGE
# =========================
@app.get("/ui", response_class=HTMLResponse)
def ui():
    try:
        with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return """
        <html>
            <head><title>DermAware Backend</title></head>
            <body style="font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px;">
                <h1>🔬 DermAware Backend</h1>
                <p>Backend is running successfully!</p>
                <h2>API Endpoints:</h2>
                <ul>
                    <li><code>GET /</code> - Health check</li>
                    <li><code>POST /classify/online</code> - Online classification</li>
                    <li><code>POST /chat</code> - AI chat</li>
                </ul>
            </body>
        </html>
        """

# =========================
# DEBUG
# =========================
@app.get("/debug")
def debug():
    return {
        "ailabtools_key": "✅ Set" if AILABTOOLS_API_KEY else "❌ Missing",
        "openai_key": "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Missing",
        "tflite_available": bool(interpreter),
        "labels_count": len(labels) if labels else 0,
        "environment": dict(os.environ) if os.getenv("DEBUG") == "true" else "hidden"
    }

# =========================
# TEST ENDPOINT
# =========================
@app.get("/test")
async def test():
    """Test that backend is working"""
    return {
        "message": "Backend is running!",
        "timestamp": "2025-01-07",
        "ailabtools_configured": bool(AILABTOOLS_API_KEY),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "ready": True
    }

# =========================
# RUN (Updated for Railway)
# =========================
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Railway sets this automatically)
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 DermAware Backend Starting...")
    print(f"🌐 Environment: {'Railway' if os.getenv('RAILWAY_ENVIRONMENT') else 'Local'}")
    print(f"📍 Port: {port}")
    print(f"🌐 Ailabtools: {'✅' if AILABTOOLS_API_KEY else '❌'}")
    print(f"🤖 OpenAI: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
    print(f"📱 TFLite: {'✅' if interpreter else '❌ (cloud mode only)'}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)