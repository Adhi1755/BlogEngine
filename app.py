"""
FastAPI REST API — Text Classification
Uses a pre-trained SVM model + TF-IDF vectorizer to predict the AG News
category of an article given its title and description.

Endpoints
---------
GET  /health   → { "status": "ok" }
POST /predict  → { "prediction": "<category>", "class_id": <int> }

Interactive docs available at:
    http://localhost:5001/docs   (Swagger UI)
    http://localhost:5001/redoc  (ReDoc)
"""

import re
import os
import logging
from contextlib import asynccontextmanager

import joblib
import nltk
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK stopwords — download once if not present
# ---------------------------------------------------------------------------
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    logger.info("Downloading NLTK stopwords …")
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Model & vectorizer paths (relative to this file)
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "SVM.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

# ---------------------------------------------------------------------------
# Label map — numeric class IDs → human-readable AG News category names
# ---------------------------------------------------------------------------
LABEL_MAP: dict = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

# ---------------------------------------------------------------------------
# Artefact store (populated during lifespan startup)
# ---------------------------------------------------------------------------
_artefacts: dict = {}


def load_artefacts() -> None:
    """Load TF-IDF vectorizer and SVM model from disk into _artefacts."""
    logger.info("Loading TF-IDF vectorizer from '%s' …", TFIDF_PATH)
    _artefacts["tfidf"] = joblib.load(TFIDF_PATH)

    logger.info("Loading SVM model from '%s' …", MODEL_PATH)
    _artefacts["model"] = joblib.load(MODEL_PATH)

    logger.info("Artefacts loaded successfully.")


# ---------------------------------------------------------------------------
# Lifespan — FastAPI's recommended startup/shutdown hook
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artefacts()          # startup
    yield
    _artefacts.clear()        # shutdown (cleanup)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Text Classification API",
    description="Classifies news articles into World / Sports / Business / Sci/Tech using a pre-trained SVM + TF-IDF pipeline.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # restrict to specific domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    title: str = ""
    description: str = ""

    @field_validator("title", "description", mode="before")
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Apple launches new iPhone",
                    "description": "The tech giant unveiled its latest smartphone at WWDC.",
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    prediction: str
    class_id: int


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def preprocess_text(title: str, description: str) -> str:
    """
    Combine title + description, then apply the same cleaning pipeline
    used during model training:
        1. Lowercase
        2. Remove non-alphabetic characters
        3. Tokenise (whitespace split)
        4. Remove English stopwords
        5. Re-join into a single string
    """
    combined = f"{title} {description}".lower()
    combined = re.sub(r"[^a-z\s]", "", combined)
    tokens   = [t for t in combined.split() if t not in STOPWORDS]
    cleaned  = " ".join(tokens)
    logger.debug("Cleaned text: %s", cleaned)
    return cleaned


def predict_category(cleaned_text: str) -> tuple[int, str]:
    """
    Vectorise with TF-IDF, run SVM inference, map to human-readable label.

    Returns
    -------
    tuple: (class_id, category_name)
    """
    tfidf    = _artefacts["tfidf"]
    model    = _artefacts["model"]
    vector   = tfidf.transform([cleaned_text])
    class_id = int(model.predict(vector)[0])
    category = LABEL_MAP.get(class_id, "Unknown")
    return class_id, category


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health():
    """Quick liveness check. Returns `{ "status": "ok" }` when the service is running."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(body: PredictRequest):
    """
    Classify a news article into one of four AG News categories.

    - **title**: article headline (optional if description is provided)
    - **description**: article body/summary (optional if title is provided)
    """
    if not body.title and not body.description:
        raise HTTPException(
            status_code=400,
            detail="Both 'title' and 'description' are empty. At least one must be provided.",
        )

    try:
        cleaned_text       = preprocess_text(body.title, body.description)
        class_id, category = predict_category(cleaned_text)

        logger.info(
            "Prediction successful | title='%s' | class_id=%d | category='%s'",
            body.title[:60], class_id, category,
        )

        return PredictResponse(prediction=category, class_id=class_id)

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="info",
    )
