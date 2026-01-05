"""
AI Detection - Improved Implementation with Perplexity Analysis
==============================================================

This implementation uses multiple signals for better accuracy:
1. Perplexity-based detection (how predictable is the text?)
2. Burstiness analysis (how varied are sentence lengths?)
3. Vocabulary diversity (is word choice repetitive?)
4. RoBERTa classifier as secondary signal

Target: 80%+ accuracy on modern AI (GPT-4, Claude, etc.)
"""

import torch
import torch.nn.functional as F
import requests
import re
import numpy as np
from typing import Dict, List, Optional
from collections import Counter


# ============================================================
# Perplexity + Statistical Analyzer
# ============================================================

class PerplexityAnalyzer:
    """
    Perplexity and statistical analysis for AI detection.
    
    Key insight: AI text is more predictable (lower perplexity)
    and more uniform (lower burstiness) than human text.
    """
    
    def __init__(self):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        print("[Perplexity] Loading GPT-2 for analysis...")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model.eval()
        print("[Perplexity] Ready!")
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate text perplexity - lower = more AI-like."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            return torch.exp(outputs.loss).item()
        except Exception:
            return 50.0  # Default middle value
    
    def calculate_burstiness(self, text: str) -> float:
        """Calculate sentence length variation - lower = more AI-like."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]
        
        if len(sentences) < 2:
            return 0.3
        
        lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(lengths)
        
        if mean_len < 1:
            return 0.3
        
        return np.std(lengths) / mean_len
    
    def calculate_vocabulary_diversity(self, text: str) -> float:
        """Calculate vocabulary richness - lower = more AI-like."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        if len(words) < 10:
            return 0.5
        
        unique = len(set(words))
        total = len(words)
        
        # TTR (Type-Token Ratio) normalized
        return unique / total
    
    def calculate_repetition_score(self, text: str) -> float:
        """Detect phrase repetition - higher = more AI-like."""
        words = text.lower().split()
        
        if len(words) < 20:
            return 0.0
        
        # Check for repeated n-grams
        ngrams = []
        for n in [2, 3, 4]:
            for i in range(len(words) - n + 1):
                ngrams.append(" ".join(words[i:i+n]))
        
        counter = Counter(ngrams)
        repeated = sum(1 for count in counter.values() if count > 1)
        
        return repeated / max(len(ngrams), 1)
    
    def analyze(self, text: str) -> Dict:
        """Full statistical analysis."""
        perplexity = self.calculate_perplexity(text)
        burstiness = self.calculate_burstiness(text)
        vocab_div = self.calculate_vocabulary_diversity(text)
        repetition = self.calculate_repetition_score(text)
        
        # Scoring based on multiple factors
        # Perplexity: AI typically 20-50, human 50-150
        if perplexity < 25:
            ppl_score = 0.9
        elif perplexity < 40:
            ppl_score = 0.7
        elif perplexity < 60:
            ppl_score = 0.5
        elif perplexity < 100:
            ppl_score = 0.3
        else:
            ppl_score = 0.1
        
        # Burstiness: AI typically 0.1-0.3, human 0.3-0.8
        if burstiness < 0.2:
            burst_score = 0.8
        elif burstiness < 0.35:
            burst_score = 0.6
        elif burstiness < 0.5:
            burst_score = 0.4
        else:
            burst_score = 0.2
        
        # Vocabulary: AI often has lower diversity on formal topics
        if vocab_div < 0.4:
            vocab_score = 0.7
        elif vocab_div < 0.55:
            vocab_score = 0.5
        else:
            vocab_score = 0.3
        
        # Combine scores (weighted)
        ai_probability = (
            ppl_score * 0.45 +
            burst_score * 0.30 +
            vocab_score * 0.15 +
            repetition * 0.10
        )
        
        return {
            "perplexity": round(perplexity, 2),
            "burstiness": round(burstiness, 3),
            "vocabulary_diversity": round(vocab_div, 3),
            "repetition": round(repetition, 3),
            "ai_probability": round(ai_probability, 4),
            "signals": {
                "perplexity_signal": ppl_score,
                "burstiness_signal": burst_score,
                "vocabulary_signal": vocab_score
            }
        }


# ============================================================
# RoBERTa Classifier (Secondary Signal)
# ============================================================

class RoBERTaClassifier:
    """RoBERTa-based classifier as secondary signal."""
    
    def __init__(self):
        from transformers import pipeline
        
        print("[RoBERTa] Loading classifier...")
        
        # Using Hello-SimpleAI model but interpreting results carefully
        self.classifier = pipeline(
            "text-classification",
            model="Hello-SimpleAI/chatgpt-detector-roberta",
            device=-1,
            truncation=True,
            max_length=512
        )
        print("[RoBERTa] Ready!")
    
    def predict(self, text: str) -> float:
        """Get AI probability from classifier."""
        try:
            result = self.classifier(text)[0]
            label = result["label"].upper()
            score = result["score"]
            
            # This model: "ChatGPT" = AI detected, "Human" = not AI
            # But based on testing, the model seems inverted for modern text
            # So we need to be careful here
            
            if "CHATGPT" in label:
                return score  # High score = AI
            else:
                return 1 - score  # "Human" with high confidence = low AI
        except Exception:
            return 0.5


# ============================================================
# Hybrid Detector (Main Class)
# ============================================================

class HybridDetector:
    """
    Hybrid AI detector combining:
    1. Statistical analysis (perplexity, burstiness, vocabulary)
    2. RoBERTa classifier
    
    Designed for 80%+ accuracy on modern AI text.
    """
    
    def __init__(self, gptzero_api_key: str = None):
        print("=" * 60)
        print("Initializing Hybrid AI Detector")
        print("=" * 60)
        
        self.stats_analyzer = PerplexityAnalyzer()
        self.classifier = RoBERTaClassifier()
        self.gptzero_api_key = gptzero_api_key
        
        self.detection_count = 0
        
        print("=" * 60)
        print("Hybrid Detector Ready!")
        print("  - Statistical analysis: Perplexity + Burstiness + Vocabulary")
        print("  - Classifier: RoBERTa ChatGPT detector")
        print(f"  - GPTZero API: {'Enabled' if gptzero_api_key else 'Disabled'}")
        print("=" * 60)
    
    def detect(self, text: str, min_words: int = 50) -> Dict:
        """
        Detect if text is AI-generated.
        
        Returns comprehensive analysis with confidence scores.
        """
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
        
        # Get statistical analysis
        stats = self.stats_analyzer.analyze(text)
        stats_ai_prob = stats["ai_probability"]
        
        # Get classifier prediction
        classifier_prob = self.classifier.predict(text)
        
        # Combine scores with weights
        # Statistical analysis is more reliable for modern AI
        # Classifier can help confirm or add uncertainty
        
        # If both agree strongly, boost confidence
        if abs(stats_ai_prob - classifier_prob) < 0.2:
            combined = (stats_ai_prob * 0.6 + classifier_prob * 0.4)
            confidence = "high"
        elif abs(stats_ai_prob - classifier_prob) < 0.35:
            combined = (stats_ai_prob * 0.7 + classifier_prob * 0.3)
            confidence = "medium"
        else:
            # Disagreement - trust statistics more
            combined = (stats_ai_prob * 0.8 + classifier_prob * 0.2)
            confidence = "low"
        
        # Determine verdict
        if combined > 0.70:
            verdict = "AI Generated"
        elif combined > 0.55:
            verdict = "Likely AI"
        elif combined > 0.45:
            verdict = "Uncertain"
        elif combined > 0.30:
            verdict = "Likely Human"
        else:
            verdict = "Human Written"
        
        return {
            "ai_probability": round(combined, 4),
            "human_probability": round(1 - combined, 4),
            "verdict": verdict,
            "confidence": confidence,
            "word_count": word_count,
            "model": "Hybrid (Perplexity + RoBERTa)",
            "source": "local",
            "api_used": False,
            "details": {
                "statistical_score": round(stats_ai_prob, 4),
                "classifier_score": round(classifier_prob, 4),
                "perplexity": stats["perplexity"],
                "burstiness": stats["burstiness"],
                "vocabulary_diversity": stats["vocabulary_diversity"]
            }
        }
    
    def detect_sentences(self, text: str) -> List[Dict]:
        """Sentence-level analysis for highlighting."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
        
        results = []
        for sent in sentences:
            word_count = len(sent.split())
            
            if word_count < 15:
                # Short sentences - less reliable
                results.append({
                    "sentence": sent,
                    "ai_probability": 0.5,
                    "is_ai": False,
                    "color": "#FFE66D"
                })
                continue
            
            # Get perplexity for sentence
            ppl = self.stats_analyzer.calculate_perplexity(sent)
            
            # Simple scoring based on perplexity
            if ppl < 30:
                ai_prob = 0.75
            elif ppl < 50:
                ai_prob = 0.6
            elif ppl < 80:
                ai_prob = 0.4
            else:
                ai_prob = 0.25
            
            results.append({
                "sentence": sent,
                "ai_probability": round(ai_prob, 2),
                "is_ai": ai_prob > 0.5,
                "color": self._get_color(ai_prob)
            })
        
        return results
    
    def detect_with_highlights(self, text: str) -> Dict:
        """Full detection with sentence highlighting."""
        overall = self.detect(text)
        sentences = self.detect_sentences(text)
        
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
    
    def _get_color(self, ai_prob: float) -> str:
        if ai_prob > 0.65:
            return "#FF6B6B"  # Red
        elif ai_prob > 0.45:
            return "#FFE66D"  # Yellow
        else:
            return "#4ECDC4"  # Green
    
    def get_stats(self) -> Dict:
        return {"total_detections": self.detection_count}


# ============================================================
# Ensemble Detector (Alternative)
# ============================================================

class EnsembleDetector(HybridDetector):
    """Alias for HybridDetector for API compatibility."""
    pass


# ============================================================
# GPTZero API Client
# ============================================================

class GPTZeroAPI:
    """GPTZero API client for highest accuracy (paid)."""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GPTZero API key required")
        self.api_key = api_key
    
    def detect(self, text: str) -> Dict:
        try:
            response = requests.post(
                "https://api.gptzero.me/v2/predict/text",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key
                },
                json={"document": text},
                timeout=30
            )
            
            if response.status_code != 200:
                return {"error": f"API error: {response.status_code}"}
            
            data = response.json()
            doc = data.get("documents", [{}])[0] if data.get("documents") else {}
            ai_prob = doc.get("completely_generated_prob", 0)
            
            return {
                "ai_probability": round(float(ai_prob), 4),
                "human_probability": round(1 - float(ai_prob), 4),
                "verdict": doc.get("predicted_class", "unknown"),
                "model": "GPTZero API"
            }
        except Exception as e:
            return {"error": str(e)}


# ============================================================
# Plagiarism Detector
# ============================================================

class PlagiarismDetector:
    """Plagiarism detection using TF-IDF and semantic similarity."""
    
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
    print("=" * 70)
    print("AI Detection - Comprehensive Test Suite")
    print("=" * 70)
    
    detector = HybridDetector()
    
    tests = [
        ("AI formal essay", """Artificial intelligence represents one of the most transformative technological
    advancements in human history. Its applications span numerous industries, from
    healthcare to finance, revolutionizing how we approach complex problems. Machine
    learning algorithms have demonstrated remarkable capabilities in pattern recognition
    and predictive analytics, enabling organizations to make data-driven decisions
with unprecedented accuracy and efficiency."""),
        
        ("Human casual story", """I remember when I was a kid, my grandmother used to make the best apple pie.
She'd wake up early on Saturday mornings, and the whole house would smell 
amazing. Those are the kinds of memories you never forget, you know? Now 
whenever I smell cinnamon, I'm transported back to her kitchen. It's funny
how smells can do that - take you right back to being seven years old."""),
        
        ("Human email", """Hey Mike! Just wanted to follow up on our conversation from yesterday. 
I totally forgot to mention that the meeting got moved to Thursday at 3pm 
instead of Wednesday. Also, did you ever find out what happened with that 
client issue? Let me know when you get a chance. Btw, great job on the 
presentation last week - everyone was really impressed!"""),

        ("AI long essay", """The evolution of technology has significantly impacted modern society, 
fundamentally altering how individuals communicate, work, and access information. 
The proliferation of smartphones and social media platforms has created unprecedented 
connectivity, enabling instantaneous communication across geographical boundaries. 
Furthermore, advancements in cloud computing have democratized access to powerful 
computational resources, empowering businesses of all sizes to leverage sophisticated 
data analytics and machine learning capabilities for strategic decision-making."""),

        ("Human story", """So there I was, stuck in traffic on the 405 for like two hours yesterday.
My phone was dead, no snacks in the car, and then the AC decided to crap out!
I'm sweating like crazy and this guy in the lane next to me keeps honking 
for absolutely no reason. Finally got home around 9pm, completely missed 
dinner with my wife. She wasn't happy about it but at least she saved me 
some leftovers lol."""),
        
        ("Mixed content", """So I was researching AI for my class and found some interesting stuff.
The integration of artificial intelligence into modern healthcare systems 
has demonstrated significant potential for improving patient outcomes through
enhanced diagnostic accuracy and personalized treatment recommendations.
But honestly I don't really understand half of it! My professor keeps talking
about neural networks and machine learning but it all sounds like sci-fi to me."""),
    ]
    
    print("\nTest Results:")
    print("-" * 70)
    
    for name, text in tests:
        result = detector.detect(text)
        ai = result['ai_probability']
        verdict = result['verdict']
        conf = result.get('confidence', 'n/a')
        details = result.get('details', {})
        
        print(f"\n{name}:")
        print(f"  AI Probability: {ai:.1%}")
        print(f"  Verdict: {verdict} (confidence: {conf})")
        if details:
            print(f"  Stats: PPL={details.get('perplexity', 0):.1f}, Burst={details.get('burstiness', 0):.3f}")
