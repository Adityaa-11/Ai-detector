"""
AI Detection - Improved Implementation
======================================

This file includes:
1. SuperAnnotate Detector (FREE, 85%+ accuracy, trained on GPT-4/Claude/Mistral)
2. GPTZero API Client (PAID, 96%+ accuracy)
3. Hybrid Detector (best value - uses free model first, API for verification)
4. Ensemble Detector (combines multiple methods)

INSTALLATION:
    pip install git+https://github.com/superannotateai/generated_text_detector.git@v1.1.0
    pip install transformers torch requests

QUICK START:
    # Free option (recommended first)
    from detection_improved import SuperAnnotateDetector
    detector = SuperAnnotateDetector()
    result = detector.detect("Your text here")
    
    # With GPTZero API (most accurate)
    from detection_improved import HybridDetector
    detector = HybridDetector(gptzero_api_key="YOUR_KEY")
    result = detector.detect("Your text here")
"""

import torch
import torch.nn.functional as F
import requests
import re
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


# ============================================================
# Method 1: SuperAnnotate Detector (FREE - 85%+ accuracy)
# ============================================================

class SuperAnnotateDetector:
    """
    SuperAnnotate AI Detector - Best FREE option available.
    
    Trained on 14 different LLMs:
    - GPT-2, GPT-3, GPT-4, ChatGPT
    - Claude (Anthropic)
    - LLaMA, LLaMA-Chat
    - Mistral, Mistral-Chat
    - Cohere, Cohere-Chat
    - MPT, MPT-Chat
    
    Performance:
    - ChatGPT: 99.2%
    - GPT-4: 98.5%
    - GPT-3: 94.5%
    - Average: 85.2%
    
    Source: https://huggingface.co/SuperAnnotate/ai-detector
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the SuperAnnotate detector.
        
        First run will download ~1.4GB model.
        """
        try:
            from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
            from generated_text_detector.utils.preprocessing import preprocessing_text
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "Please install the SuperAnnotate detector:\n"
                "pip install git+https://github.com/superannotateai/generated_text_detector.git@v1.1.0"
            )
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print("[SuperAnnotate] Loading model (first run downloads ~1.4GB)...")
        self.model = RobertaClassifier.from_pretrained("SuperAnnotate/ai-detector")
        self.tokenizer = AutoTokenizer.from_pretrained("SuperAnnotate/ai-detector")
        self.preprocess = preprocessing_text
        
        if self.device == "cuda":
            self.model = self.model.cuda()
        
        self.model.eval()
        print(f"[SuperAnnotate] Ready on {self.device}!")
    
    def detect(self, text: str) -&gt; Dict:
        """
        Detect if text is AI-generated.
        
        Args:
            text: Text to analyze (minimum 50 words recommended)
            
        Returns:
            Dict with ai_probability, verdict, confidence
        """
        word_count = len(text.split())
        if word_count &lt; 20:
            return {
                "ai_probability": 0.5,
                "human_probability": 0.5,
                "verdict": "Text too short",
                "confidence": "none",
                "error": f"Need 20+ words, got {word_count}"
            }
        
        # Preprocess
        processed = self.preprocess(text)
        
        # Tokenize
        tokens = self.tokenizer.encode_plus(
            processed,
            add_special_tokens=True,
            max_length=512,
            padding='longest',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        
        if self.device == "cuda":
            tokens = {k: v.cuda() for k, v in tokens.items()}
        
        # Predict
        with torch.no_grad():
            _, logits = self.model(**tokens)
        
        ai_prob = F.sigmoid(logits).squeeze().item()
        
        return {
            "ai_probability": round(ai_prob, 4),
            "human_probability": round(1 - ai_prob, 4),
            "verdict": self._get_verdict(ai_prob),
            "confidence": self._get_confidence(ai_prob),
            "word_count": word_count,
            "model": "SuperAnnotate/ai-detector"
        }
    
    def detect_long(self, text: str, chunk_size: int = 400) -&gt; Dict:
        """
        Detect long text by chunking and averaging.
        
        Better for documents &gt;500 words.
        """
        words = text.split()
        if len(words) &lt;= chunk_size:
            return self.detect(text)
        
        # Chunk with overlap
        chunks = []
        stride = chunk_size // 2
        for i in range(0, len(words), stride):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) &gt;= 50:
                chunks.append(chunk)
        
        if not chunks:
            return self.detect(text)
        
        # Get predictions for each chunk
        scores = []
        for chunk in chunks:
            result = self.detect(chunk)
            if "error" not in result:
                scores.append(result["ai_probability"])
        
        if not scores:
            return self.detect(text[:1000])  # Fallback
        
        avg_score = float(np.mean(scores))
        
        return {
            "ai_probability": round(avg_score, 4),
            "human_probability": round(1 - avg_score, 4),
            "verdict": self._get_verdict(avg_score),
            "confidence": self._get_confidence(avg_score),
            "word_count": len(words),
            "chunks_analyzed": len(chunks),
            "score_range": [round(min(scores), 4), round(max(scores), 4)],
            "model": "SuperAnnotate/ai-detector"
        }
    
    def _get_verdict(self, prob: float) -&gt; str:
        if prob &gt; 0.75:
            return "AI Generated"
        elif prob &gt; 0.55:
            return "Likely AI"
        elif prob &gt; 0.45:
            return "Uncertain"
        elif prob &gt; 0.25:
            return "Likely Human"
        else:
            return "Human Written"
    
    def _get_confidence(self, prob: float) -&gt; str:
        dist = abs(prob - 0.5)
        if dist &gt; 0.3:
            return "high"
        elif dist &gt; 0.15:
            return "medium"
        else:
            return "low"


# ============================================================
# Method 2: GPTZero API (PAID - 96%+ accuracy)
# ============================================================

class GPTZeroAPI:
    """
    GPTZero API Client.
    
    Accuracy: 96.5% on mixed documents
    False positive rate: &lt;1%
    
    Pricing (as of Jan 2025):
    - 300K words/mo: $45
    - 1M words/mo: $135
    - 2M words/mo: $250
    - 5M words/mo: $550
    - 10M words/mo: $1000
    - Overage: $150 per 1M words
    
    Get API key: https://app.gptzero.me/api
    """
    
    BASE_URL = "https://api.gptzero.me/v2/predict/text"
    
    def __init__(self, api_key: str):
        """
        Initialize GPTZero API client.
        
        Args:
            api_key: Your GPTZero API key
        """
        if not api_key:
            raise ValueError("GPTZero API key required. Get one at https://app.gptzero.me/api")
        
        self.api_key = api_key
        self.requests_made = 0
        self.words_used = 0
    
    def detect(self, text: str) -&gt; Dict:
        """
        Detect AI-generated text using GPTZero API.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with classification, confidence, sentence highlights
        """
        word_count = len(text.split())
        
        try:
            response = requests.post(
                self.BASE_URL,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key
                },
                json={"document": text},
                timeout=30
            )
            
            self.requests_made += 1
            self.words_used += word_count
            
            if response.status_code == 401:
                return {"error": "Invalid API key"}
            elif response.status_code == 429:
                return {"error": "Rate limit exceeded"}
            elif response.status_code != 200:
                return {"error": f"API error: {response.status_code}"}
            
            data = response.json()
            
            # Parse response
            doc = data.get("documents", [{}])[0] if data.get("documents") else {}
            
            # Get probabilities
            probs = doc.get("class_probabilities", {})
            ai_prob = probs.get("ai", doc.get("completely_generated_prob", 0))
            
            # Get classification
            classification = doc.get("predicted_class", "unknown")
            if classification == "AI_ONLY":
                verdict = "AI Generated"
            elif classification == "MIXED":
                verdict = "Mixed AI/Human"
            elif classification == "HUMAN_ONLY":
                verdict = "Human Written"
            else:
                verdict = classification
            
            return {
                "ai_probability": round(float(ai_prob), 4),
                "human_probability": round(1 - float(ai_prob), 4),
                "verdict": verdict,
                "confidence": doc.get("confidence_category", "unknown"),
                "classification": classification,
                "sentences": doc.get("sentences", []),
                "word_count": word_count,
                "model": "GPTZero API"
            }
            
        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def get_usage(self) -&gt; Dict:
        """Get API usage statistics."""
        return {
            "requests_made": self.requests_made,
            "words_used": self.words_used,
            "estimated_cost": self._estimate_cost()
        }
    
    def _estimate_cost(self) -&gt; str:
        """Estimate cost based on usage."""
        if self.words_used &lt;= 300_000:
            return "$45 (300K plan)"
        elif self.words_used &lt;= 1_000_000:
            return "$135 (1M plan)"
        elif self.words_used &lt;= 2_000_000:
            return "$250 (2M plan)"
        elif self.words_used &lt;= 5_000_000:
            return "$550 (5M plan)"
        else:
            return f"${550 + (self.words_used - 5_000_000) / 1_000_000 * 150:.2f}"


# ============================================================
# Method 3: Hybrid Detector (Best Value)
# ============================================================

class HybridDetector:
    """
    Hybrid detector combining free local model with GPTZero API.
    
    Strategy:
    1. Run text through SuperAnnotate model (free)
    2. If result is uncertain (40-60%), verify with GPTZero API
    3. Reduces API costs by ~70%
    
    Example savings:
    - 10,000 scans fully on API: ~$550/mo
    - 10,000 scans hybrid (~30% API): ~$165/mo
    """
    
    def __init__(
        self,
        gptzero_api_key: str = None,
        uncertainty_threshold: tuple = (0.40, 0.60)
    ):
        """
        Initialize hybrid detector.
        
        Args:
            gptzero_api_key: Optional GPTZero API key for verification
            uncertainty_threshold: Range where API verification is used
        """
        print("=" * 50)
        print("Initializing Hybrid AI Detector")
        print("=" * 50)
        
        self.local = SuperAnnotateDetector()
        self.gptzero = GPTZeroAPI(gptzero_api_key) if gptzero_api_key else None
        self.threshold = uncertainty_threshold
        
        self.stats = {
            "total_requests": 0,
            "local_only": 0,
            "api_verified": 0
        }
        
        print("=" * 50)
        print(f"Hybrid Detector Ready!")
        print(f"  - Local model: SuperAnnotate (free)")
        print(f"  - API: {'GPTZero (enabled)' if self.gptzero else 'None (disabled)'}")
        print(f"  - Verification threshold: {uncertainty_threshold}")
        print("=" * 50)
    
    def detect(
        self,
        text: str,
        force_api: bool = False,
        prefer_accuracy: bool = False
    ) -&gt; Dict:
        """
        Detect AI-generated text with smart API usage.
        
        Args:
            text: Text to analyze
            force_api: Always use GPTZero API
            prefer_accuracy: Use API for medium confidence too
            
        Returns:
            Dict with detection results
        """
        self.stats["total_requests"] += 1
        
        # Step 1: Local model
        local_result = self.local.detect(text)
        
        if "error" in local_result:
            return local_result
        
        ai_prob = local_result["ai_probability"]
        
        # Determine if uncertain
        low, high = self.threshold
        is_uncertain = low &lt; ai_prob &lt; high
        is_medium_conf = 0.35 &lt; ai_prob &lt; 0.65
        
        # Step 2: Decide if API needed
        need_api = force_api or is_uncertain or (prefer_accuracy and is_medium_conf)
        
        if not need_api or not self.gptzero:
            self.stats["local_only"] += 1
            return {
                **local_result,
                "source": "local",
                "api_used": False,
                "needs_verification": is_uncertain and not self.gptzero
            }
        
        # Step 3: API verification
        self.stats["api_verified"] += 1
        api_result = self.gptzero.detect(text)
        
        if "error" in api_result:
            # API failed, return local with warning
            return {
                **local_result,
                "source": "local",
                "api_used": False,
                "api_error": api_result["error"]
            }
        
        # Return API result (more accurate)
        return {
            "ai_probability": api_result["ai_probability"],
            "human_probability": api_result["human_probability"],
            "verdict": api_result["verdict"],
            "confidence": api_result["confidence"],
            "source": "gptzero",
            "api_used": True,
            "local_result": {
                "ai_probability": local_result["ai_probability"],
                "verdict": local_result["verdict"]
            },
            "word_count": local_result.get("word_count", 0)
        }
    
    def detect_batch(self, texts: List[str], force_api: bool = False) -&gt; List[Dict]:
        """Detect multiple texts."""
        return [self.detect(text, force_api=force_api) for text in texts]
    
    def get_stats(self) -&gt; Dict:
        """Get usage statistics."""
        api_usage = self.gptzero.get_usage() if self.gptzero else {}
        api_rate = self.stats["api_verified"] / max(self.stats["total_requests"], 1)
        
        return {
            **self.stats,
            "api_call_rate": f"{api_rate:.1%}",
            "api_usage": api_usage
        }


# ============================================================
# Method 4: Multi-Model Ensemble (Most Robust)
# ============================================================

class EnsembleDetector:
    """
    Ensemble detector combining multiple models.
    
    Uses:
    1. SuperAnnotate (primary, free)
    2. Simple RoBERTa (backup, different training data)
    3. Optional: GPTZero API (tie-breaker)
    
    More robust than single model - reduces false positives.
    """
    
    def __init__(self, gptzero_api_key: str = None):
        print("=" * 50)
        print("Initializing Ensemble AI Detector")
        print("=" * 50)
        
        # Primary: SuperAnnotate (best free model)
        self.superannotate = SuperAnnotateDetector()
        
        # Secondary: Simple RoBERTa (different perspective)
        self.roberta = None
        try:
            from transformers import pipeline
            print("[Ensemble] Loading secondary RoBERTa model...")
            self.roberta = pipeline(
                "text-classification",
                model="Hello-SimpleAI/chatgpt-detector-roberta",
                device=0 if torch.cuda.is_available() else -1
            )
            print("[Ensemble] Secondary model loaded!")
        except Exception as e:
            print(f"[Ensemble] Secondary model failed: {e}")
        
        # Optional: GPTZero for tie-breaking
        self.gptzero = GPTZeroAPI(gptzero_api_key) if gptzero_api_key else None
        
        print("=" * 50)
        print("Ensemble Ready!")
        print("=" * 50)
    
    def detect(self, text: str) -&gt; Dict:
        """
        Detect using ensemble of models.
        
        Voting strategy:
        - SuperAnnotate: 50% weight (best overall)
        - RoBERTa: 30% weight (ChatGPT-specific)
        - GPTZero: 20% weight (if available, tie-breaker)
        """
        results = {}
        weights = {}
        
        # SuperAnnotate (primary)
        sa_result = self.superannotate.detect(text)
        if "error" not in sa_result:
            results["superannotate"] = sa_result["ai_probability"]
            weights["superannotate"] = 0.5
        
        # RoBERTa (secondary)
        if self.roberta:
            try:
                rb_result = self.roberta(text[:512])[0]
                label = rb_result["label"].upper()
                score = rb_result["score"]
                
                if "FAKE" in label or "AI" in label or label == "LABEL_1":
                    results["roberta"] = score
                else:
                    results["roberta"] = 1 - score
                weights["roberta"] = 0.3
            except:
                pass
        
        # GPTZero (tie-breaker, if available and models disagree)
        if self.gptzero and len(results) &gt;= 2:
            scores = list(results.values())
            if max(scores) - min(scores) &gt; 0.3:  # Disagreement
                gptz_result = self.gptzero.detect(text)
                if "error" not in gptz_result:
                    results["gptzero"] = gptz_result["ai_probability"]
                    weights["gptzero"] = 0.2
        
        if not results:
            return {"error": "All models failed"}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Weighted average
        final_score = sum(results[k] * weights[k] for k in results)
        
        # Verdict
        if final_score &gt; 0.70:
            verdict = "AI Generated"
            confidence = "high"
        elif final_score &gt; 0.55:
            verdict = "Likely AI"
            confidence = "medium"
        elif final_score &gt; 0.45:
            verdict = "Uncertain"
            confidence = "low"
        elif final_score &gt; 0.30:
            verdict = "Likely Human"
            confidence = "medium"
        else:
            verdict = "Human Written"
            confidence = "high"
        
        return {
            "ai_probability": round(final_score, 4),
            "human_probability": round(1 - final_score, 4),
            "verdict": verdict,
            "confidence": confidence,
            "models_used": list(results.keys()),
            "individual_scores": {k: round(v, 4) for k, v in results.items()},
            "weights": {k: round(v, 3) for k, v in weights.items()}
        }


# ============================================================
# Testing Functions
# ============================================================

def test_superannotate():
    """Test SuperAnnotate detector."""
    print("\n" + "=" * 50)
    print("Testing SuperAnnotate Detector")
    print("=" * 50)
    
    detector = SuperAnnotateDetector()
    
    # Human-like text
    human_text = """
    I remember when I was a kid, my grandmother used to make the best apple pie.
    She'd wake up early on Saturday mornings, and the whole house would smell 
    amazing. Those are the kinds of memories you never forget, you know? Now 
    whenever I smell cinnamon, I'm transported back to her kitchen. It's funny 
    how smells can do that to you.
    """
    
    # AI-like text
    ai_text = """
    Artificial intelligence represents one of the most transformative technological
    advancements in human history. Its applications span numerous industries, from
    healthcare to finance, revolutionizing how we approach complex problems. Machine
    learning algorithms have demonstrated remarkable capabilities in pattern recognition
    and predictive analytics, enabling organizations to make data-driven decisions
    with unprecedented accuracy and efficiency.
    """
    
    print("\n[Human-like text]")
    result = detector.detect(human_text)
    print(f"  AI Probability: {result['ai_probability']:.1%}")
    print(f"  Verdict: {result['verdict']} ({result['confidence']})")
    
    print("\n[AI-like text]")
    result = detector.detect(ai_text)
    print(f"  AI Probability: {result['ai_probability']:.1%}")
    print(f"  Verdict: {result['verdict']} ({result['confidence']})")


def test_hybrid():
    """Test hybrid detector (requires GPTZero API key)."""
    print("\n" + "=" * 50)
    print("Testing Hybrid Detector")
    print("=" * 50)
    
    # Without API (free mode)
    detector = HybridDetector(gptzero_api_key=None)
    
    test_text = """
    The integration of artificial intelligence into modern society has precipitated
    a paradigm shift in how we conceptualize human-machine interaction. This
    technological revolution has profound implications for various sectors,
    necessitating a comprehensive reassessment of existing frameworks and
    methodologies to effectively harness the transformative potential of AI.
    """
    
    result = detector.detect(test_text)
    print(f"\nResult: {result['verdict']} ({result['ai_probability']:.1%})")
    print(f"Source: {result['source']}")
    print(f"Stats: {detector.get_stats()}")


if __name__ == "__main__":
    print("AI Detection - Improved Implementation")
    print("=" * 50)
    
    # Test SuperAnnotate (free)
    test_superannotate()
    
    # Test Hybrid (without API key = free mode)
    # test_hybrid()
    
    print("\n" + "=" * 50)
    print("Tests complete!")
    print("=" * 50)
