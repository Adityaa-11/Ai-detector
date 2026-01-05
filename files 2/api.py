"""
AI &amp; Plagiarism Detection API
==============================

FastAPI backend for the detection SaaS.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /api/detect          - Full detection (AI + Plagiarism)
    POST /api/ai-detect       - AI detection only  
    POST /api/plagiarism      - Plagiarism detection only
    GET  /api/health          - Health check
    GET  /docs                - API documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import time
import hashlib
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="AI &amp; Plagiarism Detection API",
    description="Detect AI-generated content and plagiarism with 80%+ accuracy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detectors
ai_detector = None
plagiarism_detector = None


# ============== Models ==============

class TextInput(BaseModel):
    text: str = Field(..., min_length=50, description="Text to analyze")
    check_ai: bool = Field(True, description="Run AI detection")
    check_plagiarism: bool = Field(False, description="Run plagiarism detection")
    include_highlights: bool = Field(False, description="Include sentence highlighting")


class DetectionResponse(BaseModel):
    ai_probability: Optional[float] = None
    human_probability: Optional[float] = None
    verdict: Optional[str] = None
    confidence: Optional[str] = None
    plagiarism_score: Optional[float] = None
    is_plagiarized: Optional[bool] = None
    matches: Optional[List[Dict]] = None
    sentences: Optional[List[Dict]] = None
    word_count: int
    processing_time: float
    request_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    ai_detector: bool
    plagiarism_detector: bool


class CompareInput(BaseModel):
    text1: str = Field(..., min_length=20)
    text2: str = Field(..., min_length=20)


# ============== Startup ==============

@app.on_event("startup")
async def startup():
    global ai_detector, plagiarism_detector
    
    print("=" * 50)
    print("Starting AI &amp; Plagiarism Detection API")
    print("=" * 50)
    
    try:
        from detection import EnsembleDetector, PlagiarismDetector
        
        print("\nLoading AI detector...")
        ai_detector = EnsembleDetector(
            use_roberta=True,
            use_fast_detect=False  # Set True for higher accuracy (slower)
        )
        print("AI detector loaded!")
        
        print("\nLoading plagiarism detector...")
        plagiarism_detector = PlagiarismDetector()
        print("Plagiarism detector loaded!")
        
    except Exception as e:
        print(f"\nError loading detectors: {e}")
        print("Some features may not work.")
    
    print("\n" + "=" * 50)
    print("API Ready! Visit /docs for documentation")
    print("=" * 50)


# ============== Endpoints ==============

@app.get("/")
async def root():
    return {"message": "AI &amp; Plagiarism Detection API", "docs": "/docs"}


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        ai_detector=ai_detector is not None,
        plagiarism_detector=plagiarism_detector is not None
    )


@app.post("/api/detect", response_model=DetectionResponse)
async def detect(input: TextInput):
    """Main detection endpoint."""
    start = time.time()
    request_id = hashlib.md5(f"{input.text[:50]}{time.time()}".encode()).hexdigest()[:12]
    
    word_count = len(input.text.split())
    if word_count &lt; 50:
        raise HTTPException(400, f"Need 50+ words, got {word_count}")
    
    response = DetectionResponse(
        word_count=word_count,
        processing_time=0,
        request_id=request_id
    )
    
    # AI Detection
    if input.check_ai and ai_detector:
        try:
            if input.include_highlights:
                result = ai_detector.detect_with_highlights(input.text)
                response.ai_probability = result["overall"]["ai_probability"]
                response.human_probability = result["overall"]["human_probability"]
                response.verdict = result["overall"]["verdict"]
                response.confidence = result["overall"]["confidence"]
                response.sentences = result.get("sentences", [])
            else:
                result = ai_detector.detect(input.text)
                response.ai_probability = result["ai_probability"]
                response.human_probability = result["human_probability"]
                response.verdict = result["verdict"]
                response.confidence = result["confidence"]
        except Exception as e:
            response.verdict = f"Error: {str(e)}"
    
    # Plagiarism Detection
    if input.check_plagiarism and plagiarism_detector:
        try:
            result = plagiarism_detector.check(input.text)
            response.plagiarism_score = result["plagiarism_score"]
            response.is_plagiarized = result["is_plagiarized"]
            response.matches = result["matches"]
        except Exception as e:
            response.is_plagiarized = None
    
    response.processing_time = round(time.time() - start, 3)
    return response


@app.post("/api/ai-detect")
async def ai_detect_only(input: TextInput):
    """AI detection only (faster)."""
    if not ai_detector:
        raise HTTPException(503, "AI detector not loaded")
    
    word_count = len(input.text.split())
    if word_count &lt; 50:
        raise HTTPException(400, f"Need 50+ words, got {word_count}")
    
    start = time.time()
    
    if input.include_highlights:
        result = ai_detector.detect_with_highlights(input.text)
        return {
            **result["overall"],
            "sentences": result.get("sentences", []),
            "word_count": word_count,
            "processing_time": round(time.time() - start, 3)
        }
    else:
        result = ai_detector.detect(input.text)
        result["processing_time"] = round(time.time() - start, 3)
        return result


@app.post("/api/plagiarism")
async def plagiarism_only(input: TextInput):
    """Plagiarism detection only."""
    if not plagiarism_detector:
        raise HTTPException(503, "Plagiarism detector not loaded")
    
    start = time.time()
    result = plagiarism_detector.check(input.text)
    result["processing_time"] = round(time.time() - start, 3)
    return result


@app.post("/api/compare")
async def compare_texts(input: CompareInput):
    """Compare two texts for similarity."""
    if not plagiarism_detector:
        raise HTTPException(503, "Plagiarism detector not loaded")
    
    return plagiarism_detector.compare_texts(input.text1, input.text2)


@app.post("/api/add-documents")
async def add_documents(documents: List[Dict]):
    """Add documents to plagiarism database."""
    if not plagiarism_detector:
        raise HTTPException(503, "Plagiarism detector not loaded")
    
    plagiarism_detector.add_documents(documents)
    return {"status": "success", "documents_added": len(documents)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
