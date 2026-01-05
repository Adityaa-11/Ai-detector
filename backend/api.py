"""
AI & Plagiarism Detection API
==============================

FastAPI backend for the detection SaaS.

Run with:
    cd backend
    pip install -r requirements.txt
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /api/detect          - Full detection (AI + Plagiarism)
    POST /api/ai-detect       - AI detection only  
    POST /api/plagiarism      - Plagiarism detection only
    GET  /api/health          - Health check
    GET  /docs                - API documentation
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import time
import hashlib
from datetime import datetime
import io

# Initialize FastAPI
app = FastAPI(
    title="AI & Plagiarism Detection API",
    description="Detect AI-generated content and plagiarism with 80%+ accuracy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.vercel.app",
        "*"  # For development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detectors
ai_detector = None
plagiarism_detector = None
models_loaded = False


# ============== Models ==============

class TextInput(BaseModel):
    text: str = Field(..., min_length=20, description="Text to analyze (minimum 20 characters)")
    check_ai: bool = Field(True, description="Run AI detection")
    check_plagiarism: bool = Field(False, description="Run plagiarism detection")
    include_highlights: bool = Field(False, description="Include sentence-level highlighting")


class SentenceHighlight(BaseModel):
    sentence: str
    ai_probability: float
    is_ai: bool
    color: str


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
    model_used: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    ai_detector: bool
    plagiarism_detector: bool
    models_loaded: bool


class CompareInput(BaseModel):
    text1: str = Field(..., min_length=20)
    text2: str = Field(..., min_length=20)


# ============== Startup ==============

@app.on_event("startup")
async def startup():
    global ai_detector, plagiarism_detector, models_loaded
    
    print("=" * 60)
    print("🚀 Starting AI & Plagiarism Detection API")
    print("=" * 60)
    
    try:
        # Use the IMPROVED detection with SuperAnnotate model
        # Trained on GPT-4, Claude, Mistral - 85%+ accuracy
        from detection_improved import HybridDetector, PlagiarismDetector
        
        print("\n📦 Loading AI detector (SuperAnnotate - trained on GPT-4/Claude/Mistral)...")
        ai_detector = HybridDetector(
            gptzero_api_key=None  # Set your GPTZero API key here for 96%+ accuracy
        )
        print("✅ AI detector loaded!")
        
        print("\n📦 Loading plagiarism detector...")
        plagiarism_detector = PlagiarismDetector()
        print("✅ Plagiarism detector loaded!")
        
        models_loaded = True
        
    except Exception as e:
        print(f"\n❌ Error loading detectors: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  Some features may not work. Check requirements.txt")
        models_loaded = False
    
    print("\n" + "=" * 60)
    print("🎉 API Ready!")
    print("📚 Documentation: http://localhost:8000/docs")
    print("=" * 60 + "\n")


# ============== Endpoints ==============

@app.get("/")
async def root():
    return {
        "message": "AI & Plagiarism Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        version="1.0.0",
        ai_detector=ai_detector is not None,
        plagiarism_detector=plagiarism_detector is not None,
        models_loaded=models_loaded
    )


@app.post("/api/detect", response_model=DetectionResponse)
async def detect(input: TextInput):
    """
    Main detection endpoint.
    
    Analyzes text for AI-generated content and optionally plagiarism.
    Returns probability scores, verdict, and confidence level.
    """
    start = time.time()
    request_id = hashlib.md5(f"{input.text[:50]}{time.time()}".encode()).hexdigest()[:12]
    
    word_count = len(input.text.split())
    if word_count < 50:
        raise HTTPException(
            status_code=400, 
            detail=f"Text too short. Need at least 50 words, got {word_count}. Longer text = more accurate results."
        )
    
    response = DetectionResponse(
        word_count=word_count,
        processing_time=0,
        request_id=request_id
    )
    
    # AI Detection
    if input.check_ai:
        if not ai_detector:
            raise HTTPException(
                status_code=503, 
                detail="AI detector not loaded. Please wait for model initialization."
            )
        
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
            
            response.model_used = "RoBERTa Ensemble"
        except Exception as e:
            print(f"AI Detection error: {e}")
            raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")
    
    # Plagiarism Detection
    if input.check_plagiarism:
        if not plagiarism_detector:
            raise HTTPException(
                status_code=503, 
                detail="Plagiarism detector not loaded."
            )
        
        try:
            result = plagiarism_detector.check(input.text)
            response.plagiarism_score = result["plagiarism_score"]
            response.is_plagiarized = result["is_plagiarized"]
            response.matches = result["matches"]
        except Exception as e:
            print(f"Plagiarism error: {e}")
            # Don't fail the whole request for plagiarism errors
            response.plagiarism_score = 0
            response.is_plagiarized = False
            response.matches = []
    
    response.processing_time = round(time.time() - start, 3)
    return response


@app.post("/api/ai-detect")
async def ai_detect_only(input: TextInput):
    """
    AI detection only (faster).
    
    Quick check for AI-generated content without plagiarism checking.
    """
    if not ai_detector:
        raise HTTPException(status_code=503, detail="AI detector not loaded")
    
    word_count = len(input.text.split())
    if word_count < 50:
        raise HTTPException(
            status_code=400, 
            detail=f"Need at least 50 words, got {word_count}"
        )
    
    start = time.time()
    
    try:
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
            result["word_count"] = word_count
            result["processing_time"] = round(time.time() - start, 3)
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/plagiarism")
async def plagiarism_only(input: TextInput):
    """Plagiarism detection only."""
    if not plagiarism_detector:
        raise HTTPException(status_code=503, detail="Plagiarism detector not loaded")
    
    start = time.time()
    result = plagiarism_detector.check(input.text)
    result["processing_time"] = round(time.time() - start, 3)
    return result


@app.post("/api/compare")
async def compare_texts(input: CompareInput):
    """
    Compare two texts for similarity.
    
    Useful for checking if two documents are plagiarized from each other.
    """
    if not plagiarism_detector:
        raise HTTPException(status_code=503, detail="Plagiarism detector not loaded")
    
    return plagiarism_detector.compare_texts(input.text1, input.text2)


@app.post("/api/add-documents")
async def add_documents(documents: List[Dict]):
    """
    Add documents to plagiarism database.
    
    Documents should have: id, content, title (optional), url (optional)
    """
    if not plagiarism_detector:
        raise HTTPException(status_code=503, detail="Plagiarism detector not loaded")
    
    plagiarism_detector.add_documents(documents)
    return {"status": "success", "documents_added": len(documents)}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and analyze a file.
    
    Supports: .txt, .pdf, .docx
    Max size: 10MB
    """
    # Check file size (10MB max)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")
    
    # Extract text based on file type
    filename = file.filename.lower()
    text = ""
    
    try:
        if filename.endswith('.txt'):
            text = contents.decode('utf-8')
        elif filename.endswith('.pdf'):
            # Optional: Install PyPDF2 for PDF support
            try:
                import PyPDF2
                pdf = PyPDF2.PdfReader(io.BytesIO(contents))
                text = " ".join(page.extract_text() for page in pdf.pages)
            except ImportError:
                raise HTTPException(
                    status_code=400, 
                    detail="PDF support not installed. Use .txt files or install PyPDF2."
                )
        elif filename.endswith('.docx'):
            # Optional: Install python-docx for DOCX support
            try:
                import docx
                doc = docx.Document(io.BytesIO(contents))
                text = " ".join(para.text for para in doc.paragraphs)
            except ImportError:
                raise HTTPException(
                    status_code=400, 
                    detail="DOCX support not installed. Use .txt files or install python-docx."
                )
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Use .txt, .pdf, or .docx"
            )
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not decode file. Check encoding.")
    
    if len(text.split()) < 50:
        raise HTTPException(
            status_code=400, 
            detail=f"File contains too little text ({len(text.split())} words). Need at least 50 words."
        )
    
    # Analyze the extracted text
    input_data = TextInput(text=text, check_ai=True, check_plagiarism=False, include_highlights=True)
    return await detect(input_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
