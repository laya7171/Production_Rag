import logfire
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings


BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768

_active_model = None
_model_type: str | None = None

def _probe_gemini():
    """Try one embed call to verify GEMINI is reachable. Returns model or None if"""
    #type of like health call to see if the model is working
    try: 
        model = GoogleGenerativeAIEmbeddings(model=settings.GEMINI_MODEL, api_key=settings.GEMINI_API_KEY)
        logfire.info("Probing GEMINI model...")
        model.embed_query("hello world")
        logfire.info("GEMINI model is reachable.")
        return model
    except Exception as e:
        logfire.error(f"GEMINI model is not reachable: {e}")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading fallback model...")
    model = SentenceTransformer('all-mpnet-base-v2')
    return model


def _init():
    
    global _active_model, _model_type

    if _active_model is not None:
        return 
    
    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"

def get_embedding_dim():
    """Get the dimension of the active embedding model."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Embed a bath of strings using the active model"""
    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                if_rate_limit = any(x in err for x in ["429", "rate", "quota", "resource_exhausted"])
                if if_rate_limit:
                    logfire.warning(f"GEMINI rate limit hit, retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                else:
                    logfire.error(f"GEMINI embedding error: {e}")
                    raise
        raise RuntimeError("GEMINI embedding failed after 4 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""

    _init()
    return _active_model.embed_query(query) if _model_type == "gemini" else _active_model.encode([query], show_progress_bar=False)[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = _embed_batch(batch)
        with logfire.context({"batch_index": i // BATCH_SIZE, "batch_size": len(batch)}):
            logfire.info(f"Embedded batch of {len(batch)} texts.")
        all_embeddings.extend(embeddings)

    return all_embeddings