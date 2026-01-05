# How to Improve Your AI Detection Accuracy to 80%+

## Your Options (Ranked by Effectiveness)

---

## Option 1: Use Better Pre-trained Models (FREE - Recommended First)

The model you're using (`roberta-base-openai-detector`) was trained on **GPT-2 only** - it explicitly says on HuggingFace:

&gt; "It is strongly suggested not to use it as a ChatGPT detector... this model might give inaccurate results in the case of ChatGPT-generated input."

### Best Free Models Available Now:

| Model | Trained On | Accuracy | Link |
|-------|------------|----------|------|
| **SuperAnnotate/ai-detector** 🏆 | GPT-2/3/4, Claude, LLaMA, Mistral (14 models) | **85.2% avg** (98.5% on GPT-4) | [HuggingFace](https://huggingface.co/SuperAnnotate/ai-detector) |
| Hello-SimpleAI/chatgpt-detector-roberta | ChatGPT/GPT-3.5 | ~95% on ChatGPT | [HuggingFace](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta) |
| roberta-base-openai-detector | GPT-2 only | ~95% on GPT-2 only | [HuggingFace](https://huggingface.co/openai-community/roberta-base-openai-detector) |

### SuperAnnotate Model Performance (Per Model):
```
ChatGPT:     99.2%
GPT-4:       98.5%
GPT-3:       94.5%
LLaMA-Chat:  98.0%
Mistral-Chat: 97.5%
Cohere-Chat: 90.6%
Human:       73.1% (correctly identifies humans)
Average:     85.2%
```

**This is your best FREE option** - trained on 14 different LLMs including GPT-4, Claude, and Mistral.

---

## Option 2: GPTZero API (Paid - Most Accurate)

### Pricing Breakdown:

| Plan | Words/Month | Price | Cost per 1K words |
|------|-------------|-------|-------------------|
| Starter | 300,000 | $45/mo | $0.15 |
| 1M | 1,000,000 | $135/mo | $0.135 |
| 2M | 2,000,000 | $250/mo | $0.125 |
| 5M | 5,000,000 | $550/mo | $0.11 |
| 10M | 10,000,000 | $1,000/mo | $0.10 |
| 20M | 20,000,000 | $1,850/mo | $0.0925 |

**Overage**: $150 per million words ($0.15 per 1K words)

### Real Cost Examples:
- 100 essays @ 500 words each = 50,000 words = **$7.50** (on starter plan)
- 1,000 scans @ 500 words = 500,000 words = **$67.50** (300K base + 200K overage)
- 10,000 scans @ 500 words = 5M words = **$550/mo** (5M plan)

### GPTZero API Features:
- 96.5% accuracy on mixed documents
- &lt;1% false positive rate
- Sentence-level highlighting
- Detects ChatGPT, GPT-4, GPT-5, Gemini, Claude, LLaMA
- No data storage (privacy)

### API Response:
```json
{
  "document_classification": "AI_ONLY" | "MIXED" | "HUMAN_ONLY",
  "class_probabilities": {
    "ai": 0.95,
    "human": 0.05,
    "mixed": 0.0
  },
  "confidence_category": "high" | "medium" | "low",
  "highlight_sentence_for_ai": [...]
}
```

---

## Option 3: Hybrid Approach (Best Balance) ⭐

Use **SuperAnnotate model locally** as primary (free), then verify uncertain cases with **GPTZero API**.

### Strategy:
1. Run text through SuperAnnotate model first (free)
2. If score is 40-60% (uncertain), call GPTZero API
3. This reduces API calls by ~70%

### Cost Savings:
- Without hybrid: 10,000 scans × $0.15 = $1,500/mo
- With hybrid (~30% API calls): 3,000 × $0.15 = $450/mo
- **Savings: $1,050/mo**

---

## Implementation: SuperAnnotate Model

### Installation:
```bash
pip install git+https://github.com/superannotateai/generated_text_detector.git@v1.1.0
pip install transformers torch
```

### Code:
```python
from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
from generated_text_detector.utils.preprocessing import preprocessing_text
from transformers import AutoTokenizer
import torch.nn.functional as F

class SuperAnnotateDetector:
    """
    Best free AI detector - trained on 14 LLMs including GPT-4, Claude, Mistral
    Average accuracy: 85.2%
    GPT-4 accuracy: 98.5%
    """
    
    def __init__(self):
        print("Loading SuperAnnotate AI Detector...")
        self.model = RobertaClassifier.from_pretrained("SuperAnnotate/ai-detector")
        self.tokenizer = AutoTokenizer.from_pretrained("SuperAnnotate/ai-detector")
        self.model.eval()
        print("Ready!")
    
    def detect(self, text: str) -&gt; dict:
        # Preprocess
        text = preprocessing_text(text)
        
        # Tokenize
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='longest',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            _, logits = self.model(**tokens)
        
        ai_prob = F.sigmoid(logits).squeeze(1).item()
        
        return {
            "ai_probability": round(ai_prob, 4),
            "human_probability": round(1 - ai_prob, 4),
            "verdict": self._get_verdict(ai_prob),
            "confidence": self._get_confidence(ai_prob),
            "model": "SuperAnnotate/ai-detector"
        }
    
    def _get_verdict(self, prob):
        if prob &gt; 0.7: return "AI Generated"
        if prob &gt; 0.5: return "Likely AI"
        if prob &gt; 0.3: return "Uncertain"
        return "Human Written"
    
    def _get_confidence(self, prob):
        dist = abs(prob - 0.5)
        if dist &gt; 0.3: return "high"
        if dist &gt; 0.15: return "medium"
        return "low"

# Usage
detector = SuperAnnotateDetector()
result = detector.detect("Your text here...")
print(result)
```

---

## Implementation: GPTZero API

### Get API Key:
1. Go to https://app.gptzero.me/api
2. Sign up and get your API key
3. Choose a plan starting at $45/month

### Code:
```python
import requests

class GPTZeroAPI:
    """
    GPTZero API client
    Accuracy: 96.5% on mixed documents
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.gptzero.me/v2/predict/text"
    
    def detect(self, text: str) -&gt; dict:
        response = requests.post(
            self.base_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-api-key": self.api_key
            },
            json={"document": text}
        )
        
        if response.status_code != 200:
            return {"error": response.text}
        
        data = response.json()
        
        # Parse response
        doc = data.get("documents", [{}])[0]
        
        return {
            "classification": doc.get("class_probabilities", {}),
            "verdict": doc.get("predicted_class", "unknown"),
            "confidence": doc.get("confidence_category", "unknown"),
            "ai_probability": doc.get("completely_generated_prob", 0),
            "sentences": doc.get("sentences", []),
            "raw_response": data
        }

# Usage
gptzero = GPTZeroAPI(api_key="YOUR_API_KEY")
result = gptzero.detect("Your text here...")
```

---

## Implementation: Hybrid Approach

```python
class HybridDetector:
    """
    Best of both worlds:
    - Uses free SuperAnnotate model first
    - Falls back to GPTZero API for uncertain cases
    """
    
    def __init__(self, gptzero_api_key: str = None):
        self.local = SuperAnnotateDetector()
        self.gptzero = GPTZeroAPI(gptzero_api_key) if gptzero_api_key else None
        self.api_calls = 0
    
    def detect(self, text: str, force_api: bool = False) -&gt; dict:
        # First, try local model
        local_result = self.local.detect(text)
        
        # If confident, return local result
        ai_prob = local_result["ai_probability"]
        is_uncertain = 0.35 &lt; ai_prob &lt; 0.65
        
        if not is_uncertain and not force_api:
            return {
                **local_result,
                "source": "local",
                "api_used": False
            }
        
        # If uncertain and API available, verify with GPTZero
        if self.gptzero and (is_uncertain or force_api):
            self.api_calls += 1
            api_result = self.gptzero.detect(text)
            
            return {
                "ai_probability": api_result.get("ai_probability", ai_prob),
                "verdict": api_result.get("verdict", local_result["verdict"]),
                "confidence": api_result.get("confidence", "medium"),
                "source": "gptzero_api",
                "api_used": True,
                "local_result": local_result,
                "api_result": api_result
            }
        
        # No API, return local with low confidence flag
        return {
            **local_result,
            "source": "local",
            "api_used": False,
            "needs_verification": is_uncertain
        }
    
    def get_stats(self):
        return {"total_api_calls": self.api_calls}

# Usage
detector = HybridDetector(gptzero_api_key="YOUR_KEY")  # or None for free-only
result = detector.detect("Your text here...")
```

---

## Recommendation Summary

| Approach | Cost | Accuracy | Best For |
|----------|------|----------|----------|
| **SuperAnnotate only** | FREE | 85%+ | MVP, testing, low volume |
| **GPTZero only** | $45+/mo | 96%+ | Production, high stakes |
| **Hybrid** ⭐ | ~$15-30/mo | 90%+ | Best value, smart scaling |

### Start Here:
1. **First**: Switch to SuperAnnotate model (free, much better than current)
2. **Test**: See if accuracy meets your needs
3. **If needed**: Add GPTZero API for uncertain cases or premium tier

### Quick Wins to Improve Accuracy:
1. ✅ Switch to SuperAnnotate model (biggest impact)
2. ✅ Require minimum 100+ words
3. ✅ Use sentence-level analysis, average results
4. ✅ Don't trust scores between 40-60% without verification
5. ✅ Consider ensemble of multiple models

---

## Cost Calculator

```python
def calculate_gptzero_cost(scans_per_month: int, avg_words_per_scan: int = 500):
    """Calculate monthly GPTZero API cost."""
    total_words = scans_per_month * avg_words_per_scan
    
    plans = [
        (300_000, 45),
        (1_000_000, 135),
        (2_000_000, 250),
        (5_000_000, 550),
        (10_000_000, 1000),
        (20_000_000, 1850),
    ]
    
    # Find best plan
    for words, price in plans:
        if total_words &lt;= words:
            return {"plan_words": words, "price": price, "your_words": total_words}
    
    # Over 20M, calculate overage
    base = 1850
    overage_words = total_words - 20_000_000
    overage_cost = (overage_words / 1_000_000) * 150
    return {"plan_words": 20_000_000, "price": base + overage_cost, "your_words": total_words}

# Examples
print(calculate_gptzero_cost(100))    # 100 scans = $45
print(calculate_gptzero_cost(1000))   # 1000 scans = $45 (500K words fits in 300K? No, need 1M plan = $135)
print(calculate_gptzero_cost(10000))  # 10000 scans = $550 (5M words)
```
