# AI Detection Backend

Python FastAPI backend for the AI & Plagiarism Detection service.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn api:app --reload --port 8000
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Detection Methods

### RoBERTa Classifier (Primary)
- **Accuracy**: 94-96%
- **Speed**: Fast (~100ms)
- **Model**: `roberta-base-openai-detector`

### Fast-DetectGPT (Optional)
- **Accuracy**: 85-90%
- **Speed**: Medium (~500ms)
- **Zero-shot**: No training needed

### Ensemble (Combined)
- Combines RoBERTa (70% weight) + Fast-DetectGPT (30% weight)
- Best balance of accuracy and speed

## Configuration

Edit `api.py` to change models:

```python
ai_detector = EnsembleDetector(
    use_roberta=True,      # Primary detector
    use_fast_detect=True,  # Enable for higher accuracy
    roberta_model="openai" # or "chatgpt", "fakespot"
)
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/detect` | Full detection |
| `POST /api/ai-detect` | AI only |
| `POST /api/plagiarism` | Plagiarism only |
| `POST /api/compare` | Compare texts |
| `POST /api/upload` | File upload |
| `GET /api/health` | Health check |

## Deployment

### Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Memory Requirements

- **RoBERTa only**: ~2GB RAM
- **With Fast-DetectGPT**: ~4GB RAM
- **GPU recommended** for production

