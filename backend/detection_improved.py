"""
AI Detection - Lightweight Implementation
==========================================

Uses a single RoBERTa model to reduce memory usage.
Target: 75-80% accuracy with low memory footprint.
"""

import torch
import re
import numpy as np
from typing import Dict, List


# ============================================================
# Lightweight RoBERTa Detector
# ============================================================

class LightweightDetector:
    """
    Single-model RoBERTa detector optimized for low memory.
    Uses roberta-base-openai-detector for AI detection.
    """
    
    def __init__(self):
        from transformers import pipeline
        
        print("[Detector] Loading lightweight model...")
        
        # Use only one model to save memory
        self.classifier = pipeline(
            "text-classification",
            model="roberta-base-openai-detector",
            device=-1,  # CPU only
            truncation=True,
            max_length=512
        )
        
        print("[Detector] Ready!")
    
    def detect(self, text: str) -> Dict:
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
            # Process text in chunks if too long
            chunks = self._chunk_text(text, max_words=400)
            scores = []
            
            for chunk in chunks:
                result = self.classifier(chunk)[0]
                label = result["label"].upper()
                score = result["score"]
                
                # "Fake" = AI, "Real" = Human
                if "FAKE" in label:
                    scores.append(score)
                else:
                    scores.append(1 - score)
            
            ai_prob = float(np.mean(scores)) if scores else 0.5
            
            return {
                "ai_probability": round(ai_prob, 4),
                "human_probability": round(1 - ai_prob, 4),
                "verdict": self._get_verdict(ai_prob),
                "confidence": self._get_confidence(ai_prob),
                "word_count": word_count,
                "model": "RoBERTa AI Detector"
            }
            
        except Exception as e:
            print(f"[Detector] Error: {e}")
            return {
                "ai_probability": 0.5,
                "human_probability": 0.5,
                "verdict": "Error",
                "confidence": "none",
                "error": str(e)
            }
    
    def detect_sentences(self, text: str) -> List[Dict]:
        """Sentence-level detection for highlighting."""
        # Split into sentences more carefully
        sentences = self._split_sentences(text)
        
        results = []
        for sent in sentences:
            word_count = len(sent.split())
            
            # Skip very short sentences
            if word_count < 5:
                results.append({
                    "sentence": sent,
                    "ai_probability": 0.5,
                    "is_ai": False,
                    "color": "#FFE66D"
                })
                continue
            
            try:
                result = self.classifier(sent[:512])[0]
                label = result["label"].upper()
                score = result["score"]
                
                if "FAKE" in label:
                    ai_prob = score
                else:
                    ai_prob = 1 - score
                
                results.append({
                    "sentence": sent,
                    "ai_probability": round(ai_prob, 3),
                    "is_ai": ai_prob > 0.5,
                    "color": self._get_color(ai_prob)
                })
            except Exception as e:
                print(f"[Detector] Sentence error: {e}")
                results.append({
                    "sentence": sent,
                    "ai_probability": 0.5,
                    "is_ai": False,
                    "color": "#FFE66D"
                })
        
        return results
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving meaningful chunks."""
        # Split on sentence-ending punctuation
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        
        sentences = []
        for sent in raw_sentences:
            sent = sent.strip()
            if sent and len(sent) > 10:  # Skip tiny fragments
                sentences.append(sent)
        
        # If no sentences found, split by newlines or return as-is
        if not sentences:
            sentences = [s.strip() for s in text.split('\n') if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            sentences = [text]
        
        return sentences
    
    def _chunk_text(self, text: str, max_words: int = 400) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if chunk.strip():
                chunks.append(chunk)
        return chunks if chunks else [text]
    
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
            return "#FF6B6B"  # Red - AI
        elif ai_prob > 0.45:
            return "#FFE66D"  # Yellow - Uncertain
        else:
            return "#4ECDC4"  # Green - Human


# ============================================================
# Hybrid Detector (wrapper for compatibility)
# ============================================================

class HybridDetector:
    """Wrapper around LightweightDetector for API compatibility."""
    
    def __init__(self, gptzero_api_key: str = None):
        print("=" * 60)
        print("Initializing AI Detector (Lightweight Mode)")
        print("=" * 60)
        
        self.detector = LightweightDetector()
        self.detection_count = 0
        
        print("=" * 60)
        print("Detector Ready!")
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
        result = self.detector.detect(text)
        
        return {
            **result,
            "source": "local",
            "api_used": False
        }
    
    def detect_with_highlights(self, text: str) -> Dict:
        """Detect AI content with sentence-level highlighting and mixed detection."""
        # Get sentence-level results
        sentences = self.detector.detect_sentences(text)
        
        # Calculate statistics from sentences
        if sentences:
            ai_probs = [s["ai_probability"] for s in sentences]
            ai_sentences = sum(1 for s in sentences if s["ai_probability"] > 0.55)
            human_sentences = sum(1 for s in sentences if s["ai_probability"] < 0.45)
            uncertain_sentences = len(sentences) - ai_sentences - human_sentences
            
            # Overall AI probability based on sentence analysis
            overall_ai_prob = float(np.mean(ai_probs)) if ai_probs else 0.5
            
            # Determine if content is "Mixed"
            total = len(sentences)
            ai_ratio = ai_sentences / total if total > 0 else 0
            human_ratio = human_sentences / total if total > 0 else 0
            
            # Mixed detection logic
            if ai_ratio >= 0.2 and human_ratio >= 0.2:
                # Both AI and human content present in significant amounts
                verdict = "Mixed AI & Human"
                confidence = "medium"
            elif overall_ai_prob > 0.75:
                verdict = "AI Generated"
                confidence = "high" if overall_ai_prob > 0.85 else "medium"
            elif overall_ai_prob > 0.55:
                verdict = "Likely AI"
                confidence = "medium" if overall_ai_prob > 0.65 else "low"
            elif overall_ai_prob > 0.45:
                verdict = "Uncertain"
                confidence = "low"
            elif overall_ai_prob > 0.25:
                verdict = "Likely Human"
                confidence = "medium" if overall_ai_prob < 0.35 else "low"
            else:
                verdict = "Human Written"
                confidence = "high" if overall_ai_prob < 0.15 else "medium"
        else:
            # Fallback to overall detection
            overall_result = self.detect(text)
            overall_ai_prob = overall_result.get("ai_probability", 0.5)
            verdict = overall_result.get("verdict", "Unknown")
            confidence = overall_result.get("confidence", "low")
            ai_sentences = 0
            human_sentences = 0
        
        return {
            "overall": {
                "ai_probability": round(overall_ai_prob, 4),
                "human_probability": round(1 - overall_ai_prob, 4),
                "verdict": verdict,
                "confidence": confidence,
                "ai_sentences": ai_sentences if sentences else 0,
                "human_sentences": human_sentences if sentences else 0,
                "total_sentences": len(sentences) if sentences else 0
            },
            "sentences": sentences,
            "word_count": len(text.split())
        }
    
    def get_stats(self) -> Dict:
        return {"total_detections": self.detection_count}


# Alias for compatibility
class EnsembleDetector(HybridDetector):
    pass


# ============================================================
# Lightweight Plagiarism Detector
# ============================================================

class PlagiarismDetector:
    """Minimal plagiarism detector - just TF-IDF, no heavy models."""
    
    def __init__(self, embedding_model: str = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        print("[Plagiarism] Initializing (lightweight mode)...")
        
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            max_features=5000
        )
        
        self.documents = []
        self.tfidf_matrix = None
        
        print("[Plagiarism] Ready!")
    
    def add_documents(self, documents: List[Dict]):
        self.documents.extend(documents)
        contents = [d['content'] for d in self.documents]
        self.tfidf_matrix = self.tfidf.fit_transform(contents)
    
    def check(self, text: str, threshold: float = 0.3) -> Dict:
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not self.documents:
            return {
                "plagiarism_score": 0,
                "is_plagiarized": False,
                "matches": []
            }
        
        text_tfidf = self.tfidf.transform([text])
        sims = cosine_similarity(text_tfidf, self.tfidf_matrix)[0]
        
        matches = []
        for i in range(len(self.documents)):
            if sims[i] > threshold:
                matches.append({
                    "source_id": self.documents[i].get("id", str(i)),
                    "source_title": self.documents[i].get("title", "Unknown"),
                    "similarity": round(float(sims[i]), 4)
                })
        
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        max_score = float(max(sims)) if len(sims) > 0 else 0
        
        return {
            "plagiarism_score": round(max_score * 100, 1),
            "is_plagiarized": max_score > threshold,
            "matches": matches[:10]
        }
    
    def compare_texts(self, text1: str, text2: str) -> Dict:
        """Compare two texts for similarity."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create a fresh vectorizer for comparison
        tfidf_temp = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
        tfidf_matrix = tfidf_temp.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return {
            "similarity": round(float(similarity), 4),
            "similarity_percentage": round(float(similarity) * 100, 1),
            "is_similar": similarity > 0.3
        }
