"""
AI Detection - Improved Implementation with SuperAnnotate
=========================================================

Uses SuperAnnotate model (85%+ accuracy on GPT-4/Claude/Mistral)
Falls back to perplexity-based detection if needed.

Target: 85%+ accuracy on modern AI
"""

import torch
import torch.nn.functional as F
import requests
import re
import numpy as np
from typing import Dict, List, Optional


# ============================================================
# SuperAnnotate Detector (85%+ accuracy)
# ============================================================

class SuperAnnotateDetector:
    """
    SuperAnnotate AI Detector - Best FREE option.
    
    Trained on 14 LLMs including GPT-4, Claude, Mistral.
    Average accuracy: 85.2%
    GPT-4 accuracy: 98.5%
    
    Source: https://huggingface.co/SuperAnnotate/ai-detector
    """
    
    def __init__(self, device: str = None):
        self.available = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
            from generated_text_detector.utils.preprocessing import preprocessing_text
            from transformers import AutoTokenizer
            
            print("[SuperAnnotate] Loading model (85%+ accuracy)...")
            self.model = RobertaClassifier.from_pretrained("SuperAnnotate/ai-detector")
            self.tokenizer = AutoTokenizer.from_pretrained("SuperAnnotate/ai-detector")
            self.preprocess = preprocessing_text
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self.model.eval()
            self.available = True
            print(f"[SuperAnnotate] Ready on {self.device}!")
            
        except Exception as e:
            print(f"[SuperAnnotate] Not available: {e}")
            self.available = False
    
    def detect(self, text: str) -> Dict:
        if not self.available:
            return {"error": "SuperAnnotate not available"}
        
        word_count = len(text.split())
        if word_count < 20:
            return {
                "ai_probability": 0.5,
                "human_probability": 0.5,
                "verdict": "Text too short",
                "confidence": "none",
                "error": f"Need 20+ words, got {word_count}"
            }
        
        try:
            processed = self.preprocess(text)
            
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
        except Exception as e:
            return {"error": str(e)}
    
    def detect_sentences(self, text: str) -> List[Dict]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
        
        results = []
        for sent in sentences:
            word_count = len(sent.split())
            
            if word_count < 15 or not self.available:
                results.append({
                    "sentence": sent,
                    "ai_probability": 0.5,
                    "is_ai": False,
                    "color": "#FFE66D"
                })
                continue
            
            detection = self.detect(sent)
            ai_prob = detection.get("ai_probability", 0.5)
            
            results.append({
                "sentence": sent,
                "ai_probability": round(ai_prob, 2),
                "is_ai": ai_prob > 0.5,
                "color": self._get_color(ai_prob)
            })
        
        return results
    
    def _get_verdict(self, prob: float) -> str:
        if prob > 0.75:
            return "AI Generated"
        elif prob > 0.55:
            return "Likely AI"
        elif prob > 0.45:
            return "Uncertain"
        elif prob > 0.25:
            return "Likely Human"
        else:
            return "Human Written"
    
    def _get_confidence(self, prob: float) -> str:
        dist = abs(prob - 0.5)
        if dist > 0.3:
            return "high"
        elif dist > 0.15:
            return "medium"
        else:
            return "low"
    
    def _get_color(self, ai_prob: float) -> str:
        if ai_prob > 0.65:
            return "#FF6B6B"  # Red
        elif ai_prob > 0.45:
            return "#FFE66D"  # Yellow
        else:
            return "#4ECDC4"  # Green


# ============================================================
# Fallback RoBERTa Detector
# ============================================================

class RoBERTaFallback:
    """Fallback detector using RoBERTa if SuperAnnotate fails."""
    
    def __init__(self):
        from transformers import pipeline
        
        print("[RoBERTa Fallback] Loading...")
        self.classifier = pipeline(
            "text-classification",
            model="roberta-base-openai-detector",
            device=-1,
            truncation=True,
            max_length=512
        )
        print("[RoBERTa Fallback] Ready!")
    
    def detect(self, text: str) -> Dict:
        word_count = len(text.split())
        if word_count < 20:
            return {
                "ai_probability": 0.5,
                "human_probability": 0.5,
                "verdict": "Text too short",
                "confidence": "none"
            }
        
        try:
            result = self.classifier(text[:512])[0]
            label = result["label"].upper()
            score = result["score"]
            
            # "Fake" = AI, "Real" = Human
            if "FAKE" in label:
                ai_prob = score
            else:
                ai_prob = 1 - score
            
            return {
                "ai_probability": round(ai_prob, 4),
                "human_probability": round(1 - ai_prob, 4),
                "verdict": self._get_verdict(ai_prob),
                "confidence": self._get_confidence(ai_prob),
                "word_count": word_count,
                "model": "RoBERTa Fallback"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_verdict(self, prob: float) -> str:
        if prob > 0.70: return "AI Generated"
        elif prob > 0.55: return "Likely AI"
        elif prob > 0.45: return "Uncertain"
        elif prob > 0.30: return "Likely Human"
        else: return "Human Written"
    
    def _get_confidence(self, prob: float) -> str:
        dist = abs(prob - 0.5)
        if dist > 0.3: return "high"
        elif dist > 0.15: return "medium"
        else: return "low"


# ============================================================
# Hybrid Detector (Main Class)
# ============================================================

class HybridDetector:
    """
    Hybrid AI detector using SuperAnnotate as primary (85%+ accuracy).
    Falls back to RoBERTa if SuperAnnotate unavailable.
    """
    
    def __init__(self, gptzero_api_key: str = None):
        print("=" * 60)
        print("Initializing Hybrid AI Detector")
        print("=" * 60)
        
        # Try SuperAnnotate first (best accuracy)
        self.superannotate = SuperAnnotateDetector()
        
        # Fallback to RoBERTa
        self.fallback = None
        if not self.superannotate.available:
            print("[Hybrid] SuperAnnotate not available, loading fallback...")
            try:
                self.fallback = RoBERTaFallback()
            except Exception as e:
                print(f"[Hybrid] Fallback also failed: {e}")
        
        self.detection_count = 0
        
        print("=" * 60)
        print("Hybrid Detector Ready!")
        if self.superannotate.available:
            print("  ✓ Using SuperAnnotate (85%+ accuracy)")
        elif self.fallback:
            print("  ⚠ Using RoBERTa fallback (70% accuracy)")
        else:
            print("  ✗ No detectors available!")
        print("=" * 60)
    
    def detect(self, text: str, min_words: int = 50) -> Dict:
        word_count = len(text.split())
        
        if word_count < min_words:
            return {
                "ai_probability": 0.5,
                "human_probability": 0.5,
                "verdict": "Text too short",
                "confidence": "none",
                "word_count": word_count,
                "error": f"Need at least {min_words} words, got {word_count}"
            }
        
        self.detection_count += 1
        
        # Use SuperAnnotate if available
        if self.superannotate.available:
            result = self.superannotate.detect(text)
            if "error" not in result:
                return {
                    **result,
                    "source": "superannotate",
                    "api_used": False
                }
        
        # Fallback to RoBERTa
        if self.fallback:
            result = self.fallback.detect(text)
            if "error" not in result:
                return {
                    **result,
                    "source": "roberta_fallback",
                    "api_used": False
                }
        
        return {
            "ai_probability": 0.5,
            "human_probability": 0.5,
            "verdict": "Detection unavailable",
            "confidence": "none",
            "error": "No detection models available"
        }
    
    def detect_with_highlights(self, text: str) -> Dict:
        overall = self.detect(text)
        
        # Get sentence-level analysis
        if self.superannotate.available:
            sentences = self.superannotate.detect_sentences(text)
        else:
            sentences = []
        
        return {
            "overall": {
                "ai_probability": overall.get("ai_probability", 0.5),
                "human_probability": overall.get("human_probability", 0.5),
                "verdict": overall.get("verdict", "Unknown"),
                "confidence": overall.get("confidence", "low")
            },
            "sentences": sentences,
            "word_count": overall.get("word_count", len(text.split()))
        }
    
    def get_stats(self) -> Dict:
        return {"total_detections": self.detection_count}


# Alias for API compatibility
class EnsembleDetector(HybridDetector):
    pass


# ============================================================
# Plagiarism Detector
# ============================================================

class PlagiarismDetector:
    """Plagiarism detection using semantic similarity."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sentence_transformers import SentenceTransformer
        
        print("[Plagiarism] Initializing...")
        
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            max_features=10000
        )
        
        print(f"[Plagiarism] Loading {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)
        
        self.documents = []
        self.tfidf_matrix = None
        self.embeddings = None
        
        print("[Plagiarism] Ready!")
    
    def add_documents(self, documents: List[Dict]):
        self.documents.extend(documents)
        contents = [d['content'] for d in self.documents]
        
        self.tfidf_matrix = self.tfidf.fit_transform(contents)
        self.embeddings = self.embedder.encode(contents, convert_to_tensor=True)
    
    def check(self, text: str, threshold: float = 0.3) -> Dict:
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import util
        
        if not self.documents:
            return {
                "plagiarism_score": 0,
                "is_plagiarized": False,
                "matches": []
            }
        
        text_tfidf = self.tfidf.transform([text])
        tfidf_sims = cosine_similarity(text_tfidf, self.tfidf_matrix)[0]
        
        text_emb = self.embedder.encode(text, convert_to_tensor=True)
        semantic_sims = util.cos_sim(text_emb, self.embeddings)[0].cpu().numpy()
        
        matches = []
        for i in range(len(self.documents)):
            max_sim = max(tfidf_sims[i], semantic_sims[i])
            if max_sim > threshold:
                matches.append({
                    "source_id": self.documents[i].get("id", str(i)),
                    "source_title": self.documents[i].get("title", "Unknown"),
                    "similarity": round(float(max_sim), 4)
                })
        
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        max_score = max(max(tfidf_sims), max(semantic_sims))
        
        return {
            "plagiarism_score": round(float(max_score) * 100, 1),
            "is_plagiarized": max_score > threshold,
            "matches": matches[:10]
        }


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("Testing AI Detection...")
    detector = HybridDetector()
    
    test_text = """
    Artificial intelligence represents one of the most transformative 
    technological advancements in human history. Its applications span 
    numerous industries, from healthcare to finance, revolutionizing 
    how we approach complex problems.
    """
    
    result = detector.detect(test_text)
    print(f"Result: {result}")
