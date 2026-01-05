# ContentVerify - AI & Plagiarism Detection SaaS

A production-ready AI content detection and plagiarism checking platform with **80%+ accuracy**.

![ContentVerify](public/placeholder-logo.svg)

## Features

- 🤖 **AI Content Detection** - Detect ChatGPT, GPT-4, Claude, and other AI-generated text
- 📝 **Plagiarism Checking** - Compare against document databases
- 📊 **Sentence-level Analysis** - See exactly which parts are AI-generated
- 📁 **File Upload Support** - Analyze TXT, PDF, and DOCX files
- 🚀 **Fast Processing** - Results in seconds
- 📱 **Responsive Design** - Works on all devices

## Tech Stack

### Frontend
- **Next.js 16** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS 4** - Styling
- **Radix UI** - Accessible components
- **Lucide Icons** - Beautiful icons

### Backend
- **Python 3.9+** - Backend language
- **FastAPI** - High-performance API framework
- **PyTorch** - Deep learning
- **Transformers** - Pre-trained models
- **Sentence Transformers** - Semantic similarity

## Detection Methods

Based on academic research for 80%+ accuracy:

| Method | Accuracy | Speed |
|--------|----------|-------|
| RoBERTa Classifier | 94-96% | Fast |
| Fast-DetectGPT | 85-90% | Medium |
| Ensemble (Combined) | 80-90%+ | Fast |

## Quick Start

### Prerequisites

- Node.js 18+ and pnpm
- Python 3.9+
- 8GB+ RAM recommended (for ML models)

### 1. Clone & Install Frontend

```bash
# Install frontend dependencies
pnpm install

# Create environment file
cp .env.example .env.local
```

### 2. Setup Python Backend

```bash
# Navigate to backend
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Both Servers

**Terminal 1 - Backend (Python):**
```bash
cd backend
uvicorn api:app --reload --port 8000
```

**Terminal 2 - Frontend (Next.js):**
```bash
pnpm dev
```

### 4. Open the App

Visit [http://localhost:3000](http://localhost:3000)

## API Endpoints

The Python backend provides these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect` | POST | Full AI + plagiarism detection |
| `/api/ai-detect` | POST | AI detection only |
| `/api/plagiarism` | POST | Plagiarism detection only |
| `/api/compare` | POST | Compare two texts |
| `/api/upload` | POST | Upload and analyze file |
| `/api/health` | GET | Health check |
| `/docs` | GET | API documentation (Swagger) |

### Example API Request

```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text to analyze here...",
    "check_ai": true,
    "check_plagiarism": false,
    "include_highlights": true
  }'
```

### Example Response

```json
{
  "ai_probability": 0.87,
  "human_probability": 0.13,
  "verdict": "AI Generated",
  "confidence": "high",
  "word_count": 150,
  "processing_time": 0.523,
  "request_id": "abc123def456",
  "sentences": [
    {
      "sentence": "This is the first sentence...",
      "ai_probability": 0.92,
      "is_ai": true,
      "color": "#FF6B6B"
    }
  ]
}
```

## Deployment

### Frontend (Vercel)

1. Push to GitHub
2. Import to Vercel
3. Set `BACKEND_URL` environment variable to your deployed backend URL
4. Deploy

### Backend (Railway/Render/Fly.io)

1. Push backend folder to a separate repo
2. Deploy to your preferred platform
3. Set up GPU instance for faster inference (optional)
4. Update frontend `BACKEND_URL`

### Example Vercel + Railway Setup

```
Frontend: https://your-app.vercel.app
Backend:  https://your-api.railway.app
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BACKEND_URL` | Python backend API URL | `http://localhost:8000` |

### Backend Configuration

Edit `backend/api.py` to customize:

```python
# Enable Fast-DetectGPT for higher accuracy (slower)
ai_detector = EnsembleDetector(
    use_roberta=True,
    use_fast_detect=True  # Set to True for higher accuracy
)
```

## Accuracy Expectations

| Text Type | Expected Accuracy |
|-----------|-------------------|
| Pure ChatGPT output | 90-95% |
| GPT-4 output | 85-90% |
| Paraphrased AI | 70-80% |
| Mixed human/AI | 65-75% |
| Human written | 95%+ (low false positives) |

## Monetization Ideas

1. **Freemium Model** - Free tier with limits, paid for unlimited
2. **API Access** - Sell API credits to developers
3. **Enterprise Plans** - Bulk pricing for institutions
4. **White-label** - License to other companies

### Pricing Tiers Example

| Plan | Price | Checks/Month | Features |
|------|-------|--------------|----------|
| Free | $0 | 10 | Basic AI detection |
| Pro | $19/mo | 500 | + Plagiarism, API |
| Business | $49/mo | 2,000 | + Priority, Reports |
| Enterprise | Custom | Unlimited | + White-label, SLA |

## Development

### Project Structure

```
ai-detection-webpage/
├── app/                    # Next.js app directory
│   ├── api/               # API routes (proxy to backend)
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page
├── backend/               # Python backend
│   ├── api.py            # FastAPI server
│   ├── detection.py      # Detection algorithms
│   └── requirements.txt  # Python dependencies
├── components/            # React components
│   ├── checker-section.tsx
│   ├── results-display.tsx
│   └── ui/               # UI components
├── lib/                   # Utilities
│   ├── types.ts          # TypeScript types
│   └── utils.ts          # Helper functions
└── public/               # Static assets
```

### Adding New Detection Methods

1. Implement detector class in `backend/detection.py`
2. Add to `EnsembleDetector` with appropriate weight
3. Update API response types if needed

## Troubleshooting

### "Backend service unavailable"

Make sure the Python backend is running:
```bash
cd backend
uvicorn api:app --reload --port 8000
```

### "CUDA out of memory"

Reduce model size or use CPU:
```python
# In detection.py
self.device = "cpu"  # Force CPU
```

### "Module not found" errors

Install missing dependencies:
```bash
pip install -r requirements.txt
```

## License

MIT License - feel free to use for commercial purposes.

## Credits

- Detection methods based on academic research (see spec file)
- UI components from [shadcn/ui](https://ui.shadcn.com)
- Icons from [Lucide](https://lucide.dev)

