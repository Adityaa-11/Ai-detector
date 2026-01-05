// Types for AI & Plagiarism Detection

export interface DetectionRequest {
  text: string
  check_ai?: boolean
  check_plagiarism?: boolean
  include_highlights?: boolean
}

export interface SentenceHighlight {
  sentence: string
  ai_probability: number
  is_ai: boolean
  color: string
}

export interface PlagiarismMatch {
  source_id: string
  source_title: string
  source_url?: string
  similarity: number
  tfidf_score?: number
  semantic_score?: number
}

export interface DetectionResult {
  ai_probability: number | null
  human_probability: number | null
  verdict: string | null
  confidence: string | null
  plagiarism_score?: number | null
  is_plagiarized?: boolean | null
  matches?: PlagiarismMatch[]
  sentences?: SentenceHighlight[]
  word_count: number
  processing_time: number
  request_id: string
  model_used?: string
  error?: string
}

export interface HealthStatus {
  status: string
  version: string
  ai_detector: boolean
  plagiarism_detector: boolean
  models_loaded: boolean
}

