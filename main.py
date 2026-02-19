from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from groq import Groq
from dotenv import load_dotenv
import io
import json
import httpx
import traceback
import socket

load_dotenv()

# =========================
# AI CHAT (Groq)
# =========================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================
# AILABTOOLS CONFIG
# =========================
AILABTOOLS_API_KEY = os.getenv("AILABTOOLS_API_KEY")

# =========================
# Load TFLite model (for OFFLINE mode + PRE-SCREENING)
# =========================
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Load labels
with open("labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

app = FastAPI(title="DermAware Backend v3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (UI)
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# Thresholds
# =========================
CONFIDENCE_THRESHOLD = 0.40   # Minimum confidence to accept a result
NOT_SKIN_THRESHOLD   = 0.30   # Below this → "Not Skin"

# DermNet classes that map to "healthy" (no disease)
HEALTHY_LABELS = {
    "normal skin", "healthy", "healthy skin", "normal",
    "clear skin", "no disease"
}

# DermNet classes that are clearly not skin
NOT_SKIN_LABELS = {
    "not skin", "not_skin", "invalid", "background",
    "no skin", "other"
}

# =========================
# Utils
# =========================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def preprocess_image(image: Image.Image):
    """
    Better preprocessing for DermNet model.
    Uses LANCZOS for better quality resizing.
    """
    image       = image.resize((224, 224), Image.LANCZOS)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def normalize_label(label: str) -> str:
    """
    Better label normalization for DermNet classes.
    Returns: 'healthy' | 'not_skin' | original cleaned label
    """
    clean = label.lower().strip()

    if clean in HEALTHY_LABELS or any(h in clean for h in ["healthy", "normal", "clear"]):
        return "healthy"

    if clean in NOT_SKIN_LABELS or any(n in clean for n in ["not skin", "not_skin", "invalid"]):
        return "not_skin"

    return label.replace("-", " ").replace("_", " ").title()

# =========================
# 🆕 TFLite Pre-screen — Saves API credits!
# =========================
def tflite_prescreen(image: Image.Image) -> dict:
    """
    Run TFLite model BEFORE calling Ailabtools to avoid wasting credits
    on blurry or non-skin images.

    Returns:
        { "passed": True }  → Safe to call Ailabtools
        { "passed": False, "reason": str, "message": str, "confidence": float }
    """
    try:
        input_data = preprocess_image(image)
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])[0]

        # Log top-3 for debugging
        top3_idx = np.argsort(output_data)[::-1][:3]
        print("🔍 TFLite pre-screen top-3:")
        for idx in top3_idx:
            print(f"   [{idx}] {labels[idx]}: {output_data[idx]:.2%}")

        max_idx    = int(np.argmax(output_data))
        raw_label  = labels[max_idx]
        confidence = float(output_data[max_idx])
        normalized = normalize_label(raw_label)

        print(f"🔍 TFLite best: {raw_label} ({confidence:.2%}) → normalized: {normalized}")

        # ❌ Too blurry / low confidence → reject immediately
        if confidence < NOT_SKIN_THRESHOLD:
            return {
                "passed":     False,
                "reason":     "low_confidence",
                "message":    "Image is too blurry or unclear. Please take a clearer photo with better lighting.",
                "confidence": confidence
            }

        # ❌ Clearly not a skin image → reject
        if normalized == "not_skin":
            return {
                "passed":     False,
                "reason":     "not_skin",
                "message":    "This doesn't appear to be a skin image. Please upload a photo of the affected skin area.",
                "confidence": confidence
            }

        # ✅ Looks like skin — allow Ailabtools call
        print(f"✅ Pre-screen PASSED ({confidence:.2%}) — allowing Ailabtools call")
        return {
            "passed":     True,
            "confidence": confidence
        }

    except Exception as e:
        # Fail-open: if TFLite crashes for some reason, allow the API call
        print(f"⚠️ TFLite pre-screen error (allowing API anyway): {e}")
        return { "passed": True, "confidence": 0.0 }


def get_skin_info_from_openai(label: str):
    """
    Get popular/common name and info from Groq/LLaMA.
    Returns: most popular English term + detailed info.
    """
    try:
        print(f"🤖 Asking Groq for details about: {label}")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dermatology assistant. "
                        "IMPORTANT: Use the MOST POPULAR, COMMON English name that regular people use - NOT scientific/medical terms. "
                        "For example: 'Ringworm' NOT 'Tinea Corporis', 'Athlete's Foot' NOT 'Tinea Pedis', "
                        "'Cold Sore' NOT 'Herpes Simplex', 'Hives' NOT 'Urticaria', etc.\n\n"
                        "Provide info in JSON format:\n"
                        "- alsoKnownAs: The MOST POPULAR common English name people actually use (NOT medical/scientific term)\n"
                        "- explanation: 2-3 sentences simple definition in everyday English\n"
                        "- causes: 3-5 common causes in simple language\n"
                        "- dos: 3-5 care recommendations in simple, actionable language\n"
                        "- donts: 3-5 things to avoid in simple language\n"
                        "Return ONLY valid JSON. Use simple, everyday language throughout."
                    )
                },
                {
                    "role": "user",
                    "content": f"What is the most popular common name for '{label}' and explain this skin condition in simple terms."
                }
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data    = json.loads(content)

        return {
            "alsoKnownAs": data.get("alsoKnownAs", label),
            "explanation": data.get("explanation", "Information currently unavailable."),
            "causes":      data.get("causes", []),
            "dos":         data.get("dos", []),
            "donts":       data.get("donts", [])
        }
    except Exception as e:
        print(f"❌ Error getting Groq info: {e}")
        return {
            "alsoKnownAs": label,
            "explanation": "Information currently unavailable.",
            "causes":      [],
            "dos":         [],
            "donts":       []
        }

# =========================
# ONLINE: TFLite Pre-screen → AILABTOOLS API → Groq Info
# =========================
@app.post("/classify/online")
async def classify_online(file: UploadFile = File(...)):
    """
    Online classification flow:
      1. TFLite pre-screen  → reject blurry/non-skin (NO credit used)
      2. Ailabtools API     → only called for valid skin images
      3. Groq               → get popular name + info
    """
    try:
        print(f"📥 Received file: {file.filename}, type: {file.content_type}")

        contents = await file.read()
        print(f"📦 File size: {len(contents)} bytes")

        if len(contents) == 0:
            raise HTTPException(400, "Empty file received")

        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            print(f"✅ Valid image: {img.format} {img.size}")
        except Exception as e:
            raise HTTPException(400, f"Invalid image file: {str(e)}")

        # ─── STEP 1: TFLite Pre-screen ────────────────────────────────
        # This runs BEFORE the paid API call to save credits on bad images.
        print("🛡️ Running TFLite pre-screen to protect API credits...")
        prescreen = tflite_prescreen(img)

        if not prescreen["passed"]:
            print(
                f"🚫 Pre-screen FAILED [{prescreen['reason']}] "
                f"(confidence: {prescreen['confidence']:.2%}) "
                f"— Ailabtools NOT called ✅ credits saved!"
            )
            return JSONResponse({
                "label":         "Not Skin",
                "confidence":    prescreen["confidence"],
                "error":         prescreen["reason"],       # "not_skin" | "low_confidence"
                "error_message": prescreen["message"]
            })

        # ─── STEP 2: Call Ailabtools (only for valid skin images) ──────
        print("🌐 Pre-screen passed — calling Ailabtools API...")
        url     = "https://www.ailabapi.com/api/portrait/analysis/skin-disease-detection"
        headers = { "ailabapi-api-key": AILABTOOLS_API_KEY }
        files   = { "image": ("photo.jpg", contents, "image/jpeg") }

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.post(url, files=files, headers=headers)

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

        # Check for API-level errors
        error_code = result.get("error_code", 0)
        if error_code != 0:
            error_msg = result.get("error_msg", "Unknown error")
            print(f"⚠️ Ailabtools error_code: {error_code}, msg: {error_msg}")
            return JSONResponse({
                "label":      "Not Skin",
                "confidence": 0.0,
                "error":      "api_error",
                "error_message": error_msg
            })

        # Extract best result
        data    = result.get("data", {})
        results = data.get("results_english", {})

        if not results:
            print("⚠️ No results from Ailabtools")
            return JSONResponse({
                "label":      "Unknown",
                "confidence": 0.0,
                "error":      "no_results"
            })

        print(f"✅ Classification results: {results}")

        best_label      = max(results.items(), key=lambda x: x[1])
        scientific_name = best_label[0].replace("_", " ").title()
        confidence      = float(best_label[1])

        print(f"🏆 Best: {scientific_name} = {confidence:.2%}")
        normalized = normalize_label(scientific_name)

        # ─── STEP 3: Groq for popular name + info ──────────────────────
        if normalized not in ["healthy", "not_skin"]:
            print(f"🔍 Skin condition detected: {scientific_name} — getting popular name + info...")
            openai_info  = get_skin_info_from_openai(scientific_name)
            popular_name = openai_info["alsoKnownAs"]
            combined_label = f"{scientific_name} - also known as {popular_name}"

            return JSONResponse({
                "label":       combined_label,
                "confidence":  confidence,
                "all_results": results,
                "explanation": openai_info["explanation"],
                "causes":      openai_info["causes"],
                "dos":         openai_info["dos"],
                "donts":       openai_info["donts"]
            })
        else:
            print(f"✅ Result is {normalized}, no Groq info needed")
            return JSONResponse({
                "label":       scientific_name,
                "confidence":  confidence,
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
# OFFLINE: TFLite Direct Classifier
# =========================
@app.post("/classify/offline")
async def classify_offline(file: UploadFile = File(...)):
    """
    Offline analysis using TFLite directly (no external API calls).
    """
    try:
        print(f"📥 Offline: {file.filename}")

        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
        input_data = preprocess_image(image)

        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])[0]

        # Log top-3
        top3_idx = np.argsort(output_data)[::-1][:3]
        print("Top 3 predictions:")
        for idx in top3_idx:
            print(f"  [{idx}] {labels[idx]}: {output_data[idx]:.2%}")

        max_idx    = int(np.argmax(output_data))
        raw_label  = labels[max_idx]
        confidence = float(output_data[max_idx])
        normalized = normalize_label(raw_label)

        print(f"🔍 TFLite best: {raw_label} ({confidence:.2%})")

        if confidence < NOT_SKIN_THRESHOLD:
            return JSONResponse({ "label": "Not Skin", "confidence": confidence, "error": "low_confidence" })

        if normalized == "not_skin":
            return JSONResponse({ "label": "Not Skin", "confidence": confidence, "error": "not_skin" })

        if normalized == "healthy":
            return JSONResponse({ "label": "Healthy Skin", "confidence": confidence })

        if confidence < CONFIDENCE_THRESHOLD:
            return JSONResponse({ "label": normalized, "confidence": confidence, "warning": "low_confidence_result" })

        return JSONResponse({ "label": normalized, "confidence": confidence })

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
    """Unified endpoint that routes to online or offline."""
    print(f"📍 Mode: {mode}")
    if mode.lower() == "online":
        return await classify_online(file)
    else:
        return await classify_offline(file)


# =========================
# AI CHAT — With Conversation History
# =========================
class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        print(f"💬 Chat: {req.message[:50]}...")
        print(f"📚 History length: {len(req.history)} messages")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are DermAware's friendly dermatology assistant.\n\n"

                    "YOUR EXPERTISE:\n"
                    "- Skin health, skincare, and dermatology\n"
                    "- Skin conditions, symptoms, and general care\n"
                    "- Sun protection, moisturizing, and basic skincare routines\n"
                    "- When to see a doctor for skin issues\n"
                    "- General skin anatomy and function\n\n"

                    "CONVERSATION GUIDELINES:\n"
                    "- ALLOW natural conversation flow (greetings, thanks, acknowledgments, follow-ups)\n"
                    "- Respond naturally to casual responses like 'okay', 'thanks', 'I see', 'ahhh'\n"
                    "- Be friendly and conversational when discussing skin topics\n"
                    "- Remember conversation context - if discussing a topic, stay engaged\n"
                    "- Accept follow-up questions about topics already being discussed\n\n"

                    "TOPIC RESTRICTIONS (only enforce for NEW topic requests):\n"
                    "- REFUSE questions about: politics, sports, cooking, math, coding, general knowledge\n"
                    "- REFUSE medical advice for non-skin conditions\n"
                    "- For off-topic questions, say: \"I'm DermAware's skin health assistant and can only help "
                    "with questions about skin, skincare, and dermatology. Please ask me about skin-related topics!\"\n\n"

                    "EXAMPLES OF GOOD RESPONSES:\n"
                    "User: 'Ahh okay, I see'\n"
                    "You: 'Glad I could help! Feel free to ask if you have more questions about your skin.' ✅\n\n"
                    "User: 'Thanks!'\n"
                    "You: 'You're welcome! Let me know if you need anything else about skincare.' ✅\n\n"
                    "User: 'How do I treat it?'\n"
                    "You: [Give advice based on previous topic being discussed] ✅\n\n"
                    "User: 'Who won the election?'\n"
                    "You: 'I'm DermAware's skin health assistant...' ✅\n\n"

                    "MEDICAL DISCLAIMERS:\n"
                    "- Never diagnose - only provide general information\n"
                    "- Always recommend consulting healthcare professionals for serious concerns\n"
                    "- Keep responses helpful but appropriately cautious"
                )
            }
        ]

        for msg in req.history:
            messages.append({
                "role":    msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        messages.append({"role": "user", "content": req.message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )

        reply = response.choices[0].message.content
        print(f"✅ Reply: {reply[:100]}...")
        return {"reply": reply}

    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(500, f"Chat failed: {str(e)}")


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
                "dos":    ["Ensure good lighting", "Focus on skin area", "Keep camera steady", "Take from 6-12 inches away"],
                "donts":  ["Don't take photos of non-skin", "Avoid blurry images", "Don't include too much background", "Avoid extreme close-ups"]
            }

        if "healthy" in condition:
            return {
                "explanation": "Your skin appears healthy! Continue good skincare habits.",
                "causes": [],
                "dos":    ["Maintain skincare routine", "Stay hydrated", "Use SPF 30+ daily", "Get 7-9 hours sleep", "Exercise regularly"],
                "donts":  ["Don't skip sunscreen", "Don't over-wash (max 2x daily)", "Don't pick at skin", "Avoid harsh products", "Don't skip moisturizer"]
            }

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dermatology assistant. Use SIMPLE, EVERYDAY language.\n"
                        "Provide info in JSON format:\n"
                        "- explanation: 2-3 sentences in simple English with reminder to see doctor\n"
                        "- causes: 3-5 common causes in everyday language\n"
                        "- dos: 3-5 care recommendations in simple, actionable steps\n"
                        "- donts: 3-5 things to avoid in simple language\n"
                        "Return ONLY valid JSON."
                    )
                },
                {"role": "user", "content": f"Explain in simple terms: {req.label}"}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        print("✅ Got AI explanation")

        try:
            data = json.loads(content)
            return {
                "explanation": data.get("explanation", "No info available"),
                "causes":      data.get("causes", []),
                "dos":         data.get("dos", []),
                "donts":       data.get("donts", [])
            }
        except json.JSONDecodeError:
            return { "explanation": content, "causes": [], "dos": [], "donts": [] }

    except Exception as e:
        print(f"❌ Explain error: {e}")
        raise HTTPException(500, f"Explanation failed: {str(e)}")


@app.get("/")
def health():
    return {
        "status":  "ok",
        "service": "DermAware Backend v3.1",
        "features": [
            "TFLite Pre-screen (protects API credits)",
            "DermNet 23 Classes",
            "Scientific + Popular Names",
            "Improved Offline"
        ],
        "ailabtools":    "✅" if AILABTOOLS_API_KEY else "❌",
        "groq":          "✅" if os.getenv("GROQ_API_KEY") else "❌",
        "tflite":        "✅",
        "model_classes": len(labels),
        "thresholds": {
            "not_skin_below":   NOT_SKIN_THRESHOLD,
            "low_conf_below":   CONFIDENCE_THRESHOLD,
        },
        "endpoints": {
            "health":  "GET /",
            "online":  "POST /classify/online",
            "offline": "POST /classify/offline",
            "unified": "POST /classify",
            "chat":    "POST /chat (with conversation history)",
            "explain": "POST /explain_result",
            "ui":      "GET /ui"
        }
    }


@app.get("/ui", response_class=HTMLResponse)
def ui():
    try:
        with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<h1>UI not available</h1><p>Create static/index.html</p>"


@app.get("/debug")
def debug():
    return {
        "ailabtools_key": "✅ Set" if AILABTOOLS_API_KEY else "❌ Missing",
        "groq_key":       "✅ Set" if os.getenv("GROQ_API_KEY") else "❌ Missing",
        "labels_count":   len(labels),
        "sample_labels":  labels[:5],
        "thresholds": {
            "not_skin":   NOT_SKIN_THRESHOLD,
            "confidence": CONFIDENCE_THRESHOLD
        }
    }


@app.get("/test")
async def test():
    return {
        "message":   "Backend v3.1 is running!",
        "timestamp": "2025",
        "ailabtools_configured": bool(AILABTOOLS_API_KEY),
        "features":  [
            "tflite_prescreen",
            "dermnet_23_classes",
            "improved_offline",
            "scientific_and_popular_names"
        ],
        "ready": True
    }


# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    print("DermAware Backend v3.1 Starting...")
    print(f"Local Network: http://{local_ip}:8000")
    print(f"Localhost:    http://127.0.0.1:8000")
    print(f"Ailabtools: {'✅' if AILABTOOLS_API_KEY else '❌'}")
    print(f"Groq: {'✅' if os.getenv('GROQ_API_KEY') else '❌'}")
    print(f"TFLite: ✅ ({len(labels)} classes) — also used as online pre-screener")
    print(f"Features: TFLite Pre-screen ✅ | DermNet 23 Classes ✅ | Improved Offline ✅")
    uvicorn.run(app, host="0.0.0.0", port=8000)
