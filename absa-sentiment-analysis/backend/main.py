"""
FastAPI Backend for Aspect-Based Sentiment Analysis Chatbot
Uses BERT-based model (bert-base-uncased fine-tuned for ABSA)
Exposes /chat endpoint for the React chatbot frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import re
import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ABSA Chatbot API", version="1.0.0")

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Label mapping (matches preprocess.py)
# --------------------------------------------------------------------------
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2, "conflict": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

SENTIMENT_EMOJI = {
    "positive": "😊",
    "negative": "😞",
    "neutral": "😐",
    "conflict": "🤔",
}

SENTIMENT_COLOR = {
    "positive": "green",
    "negative": "red",
    "neutral": "gray",
    "conflict": "orange",
}

# --------------------------------------------------------------------------
# Model path (anchored to main.py location, auto-picks latest run)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_latest_model_path() -> str:
    checkpoints_dir = os.path.join(BASE_DIR, "results", "checkpoints")
    final_paths = glob.glob(os.path.join(checkpoints_dir, "*/final"))
    if not final_paths:
        raise FileNotFoundError(
            f"No 'final' model folder found under {checkpoints_dir}"
        )
    latest = sorted(final_paths)[-1]
    logger.info(f"Auto-detected model path: {latest}")
    return latest


MODEL_NAME = get_latest_model_path()

# --------------------------------------------------------------------------
# Model state (lazy-loaded)
# --------------------------------------------------------------------------
_tokenizer = None
_model = None
_device = None


def load_model():
    global _tokenizer, _model, _device
    if _model is None:
        # Validate the directory exists and contains required files
        if not os.path.isdir(MODEL_NAME):
            raise FileNotFoundError(f"Model directory not found: {MODEL_NAME}")
        if not os.path.exists(os.path.join(MODEL_NAME, "config.json")):
            raise FileNotFoundError(f"config.json missing in: {MODEL_NAME}")

        logger.info(f"Loading model from: {MODEL_NAME}")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, local_files_only=True
        )
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=4, local_files_only=True
        )
        _model.to(_device)
        _model.eval()
        logger.info("Model loaded successfully.")
    return _tokenizer, _model, _device


# --------------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------------
class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class AspectResult(BaseModel):
    aspect: str
    sentiment: str
    confidence: float
    emoji: str


class ChatResponse(BaseModel):
    reply: str
    results: Optional[List[AspectResult]] = None
    raw_sentence: Optional[str] = None


# --------------------------------------------------------------------------
# NLP helpers
# --------------------------------------------------------------------------
def extract_aspects_from_sentence(sentence: str) -> List[str]:
    """
    Simple rule-based aspect extractor:
    Looks for noun phrases by extracting nouns (words preceded by adjectives
    or standalone nouns in key positions). Falls back to chunking on
    comma/conjunctions.
    """
    restaurant_aspects = [
        "food", "service", "staff", "ambiance", "price", "menu",
        "drinks", "dessert", "atmosphere", "location", "wait", "portions",
        "quality", "taste", "value", "decor", "cleanliness", "noise",
    ]
    laptop_aspects = [
        "battery", "screen", "keyboard", "performance", "speed", "price",
        "design", "display", "speakers", "camera", "software", "os",
        "build quality", "memory", "storage", "processor", "weight",
        "touchpad", "wifi", "charger",
    ]
    all_seed_aspects = restaurant_aspects + laptop_aspects

    sentence_lower = sentence.lower()
    found = []
    for aspect in all_seed_aspects:
        if aspect in sentence_lower:
            found.append(aspect)

    # If nothing found, try extracting noun-like tokens (simple heuristic)
    if not found:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence)
        stopwords = {
            "the", "and", "but", "for", "with", "this", "that", "was",
            "were", "are", "very", "really", "quite", "just", "also",
            "have", "has", "had", "its", "our", "their", "all", "not",
        }
        found = [w.lower() for w in words if w.lower() not in stopwords][:3]

    return list(dict.fromkeys(found))  # deduplicate, preserve order


def predict_aspect_sentiment(
    sentence: str, aspect: str, tokenizer, model, device
) -> tuple:
    """Predict sentiment for a (sentence, aspect) pair using BERT."""
    text = f"{sentence} [SEP] {aspect}"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

    return ID2LABEL[pred_id], round(confidence, 3)


def parse_user_intent(text: str):
    """
    Detect if user provided a sentence + aspects, or just a sentence.
    Returns (sentence, aspects_list_or_None)
    Supports formats like:
      - "The food was great but service was slow"
      - "sentence: The food was great | aspects: food, service"
      - "Analyze: good battery but bad screen"
    """
    text = text.strip()

    # Format: sentence | aspects: x, y
    pipe_match = re.search(r"aspects?:\s*(.+)", text, re.IGNORECASE)
    if pipe_match:
        aspects_raw = pipe_match.group(1)
        aspects = [a.strip() for a in re.split(r"[,;]", aspects_raw) if a.strip()]
        sentence_part = re.sub(r"\|?\s*aspects?:.*", "", text, flags=re.IGNORECASE).strip()
        sentence_part = re.sub(r"sentence:\s*", "", sentence_part, flags=re.IGNORECASE).strip()
        return sentence_part, aspects

    # Strip leading keywords like "analyze:", "review:", etc.
    sentence = re.sub(
        r"^(analyze|review|check|evaluate|assess)[:\s]+", "", text, flags=re.IGNORECASE
    ).strip()
    return sentence, None


def build_reply(sentence: str, results: List[AspectResult]) -> str:
    """Build a natural-language chatbot reply."""
    if not results:
        return (
            "I couldn't identify any specific aspects in that sentence. "
            "Try mentioning specific features like 'food', 'service', 'battery', 'screen', etc."
        )

    lines = [f"Here's the aspect-based sentiment analysis for:\n> *\"{sentence}\"*\n"]
    for r in results:
        lines.append(
            f"• **{r.aspect.capitalize()}** — {r.sentiment.upper()} {r.emoji} "
            f"(confidence: {r.confidence * 100:.1f}%)"
        )

    dominant = max(results, key=lambda x: x.confidence)
    lines.append(
        f"\nOverall, the strongest signal is **{dominant.sentiment}** sentiment "
        f"toward **{dominant.aspect}**."
    )
    lines.append(
        "\nYou can ask me to analyze another sentence, or specify aspects like:\n"
        "`The pizza was amazing | aspects: pizza, service`"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Greeting / help handler
# --------------------------------------------------------------------------
GREETINGS = {"hi", "hello", "hey", "hola", "howdy", "yo"}
HELP_WORDS = {"help", "what", "how", "usage", "?"}

WELCOME_MSG = (
    "👋 Hello! I'm your **Aspect-Based Sentiment Analysis** chatbot.\n\n"
    "Send me any review sentence and I'll analyze the sentiment for each aspect. "
    "For example:\n\n"
    "• `The food was amazing but the service was terrible.`\n"
    "• `Great battery life, but the screen could be brighter.`\n"
    "• `The food was great | aspects: food, service, price`\n\n"
    "I support **restaurant** and **laptop** reviews. Give it a try! 🚀"
)

HELP_MSG = (
    "**How to use me:**\n\n"
    "1. **Simple sentence**: Just type a review and I'll auto-detect aspects.\n"
    "   `The coffee was excellent but a bit overpriced.`\n\n"
    "2. **With specific aspects**: Use `| aspects: x, y` to control what I analyze.\n"
    "   `Great screen but terrible keyboard | aspects: screen, keyboard, battery`\n\n"
    "**Supported sentiments**: positive 😊, negative 😞, neutral 😐, conflict 🤔\n\n"
    "**Domains**: restaurants (food, service, ambiance…) and laptops (battery, screen, keyboard…)"
)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_msg = req.messages[-1]
    if last_msg.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    user_text = last_msg.content.strip()

    # Handle greetings
    if user_text.lower() in GREETINGS or not user_text:
        return ChatResponse(reply=WELCOME_MSG)

    # Handle help
    if any(w in user_text.lower() for w in HELP_WORDS) and len(user_text) < 20:
        return ChatResponse(reply=HELP_MSG)

    # Parse intent
    sentence, explicit_aspects = parse_user_intent(user_text)

    if not sentence or len(sentence) < 5:
        return ChatResponse(
            reply="Please send a longer sentence for me to analyze. 😊"
        )

    # Load model lazily
    try:
        tokenizer, model, device = load_model()
    except Exception as e:
        logger.error(f"Model load error: {e}")
        raise HTTPException(status_code=503, detail="Model unavailable. Please try again.")

    # Extract aspects
    aspects = explicit_aspects if explicit_aspects else extract_aspects_from_sentence(sentence)

    if not aspects:
        return ChatResponse(
            reply=(
                "I couldn't find any recognizable aspects in your sentence. "
                "Try being more specific, e.g. 'The **food** was great but **service** was slow.'"
            )
        )

    # Predict
    results = []
    for aspect in aspects:
        sentiment, confidence = predict_aspect_sentiment(
            sentence, aspect, tokenizer, model, device
        )
        results.append(
            AspectResult(
                aspect=aspect,
                sentiment=sentiment,
                confidence=confidence,
                emoji=SENTIMENT_EMOJI[sentiment],
            )
        )

    reply = build_reply(sentence, results)
    return ChatResponse(reply=reply, results=results, raw_sentence=sentence)


# --------------------------------------------------------------------------
# Dev runner
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)