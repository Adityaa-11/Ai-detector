"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { AlertCircle, CheckCircle2, ShieldAlert, Clock, FileText, Sparkles } from "lucide-react"
import type { DetectionResult, SentenceHighlight } from "@/lib/types"

interface ResultsDisplayProps {
  result: DetectionResult
}

export function ResultsDisplay({ result }: ResultsDisplayProps) {
  const aiScore = Math.round((result.ai_probability ?? 0) * 100)
  const humanScore = Math.round((result.human_probability ?? 0) * 100)
  const plagiarismScore = result.plagiarism_score ?? 0

  const getVerdictColor = (verdict: string | null) => {
    switch (verdict) {
      case "AI Generated":
        return "text-red-500"
      case "Likely AI":
        return "text-orange-500"
      case "Mixed AI & Human":
        return "text-purple-500"
      case "Uncertain":
        return "text-yellow-500"
      case "Likely Human":
        return "text-blue-500"
      case "Human Written":
        return "text-green-500"
      default:
        return "text-muted-foreground"
    }
  }

  const getConfidenceBadge = (confidence: string | null) => {
    switch (confidence) {
      case "high":
        return <Badge variant="outline" className="bg-green-500/10 text-green-600 border-green-500/20">High Confidence</Badge>
      case "medium":
        return <Badge variant="outline" className="bg-yellow-500/10 text-yellow-600 border-yellow-500/20">Medium Confidence</Badge>
      case "low":
        return <Badge variant="outline" className="bg-orange-500/10 text-orange-600 border-orange-500/20">Low Confidence</Badge>
      default:
        return <Badge variant="outline">Unknown</Badge>
    }
  }

  return (
    <div className="mt-8 space-y-6 pt-6 border-t border-border animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold">Analysis Results</h3>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <span className="flex items-center gap-1">
            <FileText className="w-4 h-4" />
            {result.word_count} words
          </span>
          <span className="flex items-center gap-1">
            <Clock className="w-4 h-4" />
            {result.processing_time}s
          </span>
        </div>
      </div>

      {/* Score Cards */}
      <div className="grid md:grid-cols-3 gap-4">
        <Card className={`border-2 transition-all ${humanScore > 50 ? 'border-green-500/40 bg-green-500/5' : 'border-border'}`}>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-green-500" />
              Human Content
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-500">{humanScore}%</div>
            <Progress value={humanScore} className="mt-2 h-2 [&>div]:bg-green-500" />
          </CardContent>
        </Card>

        <Card className={`border-2 transition-all ${aiScore > 50 ? 'border-orange-500/40 bg-orange-500/5' : 'border-border'}`}>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-orange-500" />
              AI-Generated
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-orange-500">{aiScore}%</div>
            <Progress value={aiScore} className="mt-2 h-2 [&>div]:bg-orange-500" />
          </CardContent>
        </Card>

        <Card className={`border-2 transition-all ${plagiarismScore > 30 ? 'border-red-500/40 bg-red-500/5' : 'border-border'}`}>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-red-500" />
              Plagiarism
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-red-500">{Math.round(plagiarismScore)}%</div>
            <Progress value={plagiarismScore} className="mt-2 h-2 [&>div]:bg-red-500" />
          </CardContent>
        </Card>
      </div>

      {/* Verdict Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center justify-between">
            <span>Detection Summary</span>
            {getConfidenceBadge(result.confidence)}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-center py-4">
            <div className="text-center">
              <p className="text-sm text-muted-foreground mb-2">Verdict</p>
              <p className={`text-2xl font-bold ${getVerdictColor(result.verdict)}`}>
                {result.verdict}
              </p>
            </div>
          </div>

          {/* Sentence breakdown if available */}
          {result.sentences && result.sentences.length > 0 && (
            <div className="grid grid-cols-3 gap-4 text-center border-t pt-4">
              <div>
                <p className="text-2xl font-bold text-red-500">
                  {result.sentences.filter((s: SentenceHighlight) => s.ai_probability > 0.55).length}
                </p>
                <p className="text-xs text-muted-foreground">AI Sentences</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-yellow-500">
                  {result.sentences.filter((s: SentenceHighlight) => s.ai_probability >= 0.45 && s.ai_probability <= 0.55).length}
                </p>
                <p className="text-xs text-muted-foreground">Uncertain</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-green-500">
                  {result.sentences.filter((s: SentenceHighlight) => s.ai_probability < 0.45).length}
                </p>
                <p className="text-xs text-muted-foreground">Human Sentences</p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 text-sm border-t pt-4">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Detection Model</span>
              <span className="font-medium">{result.model_used || 'RoBERTa Ensemble'}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Request ID</span>
              <span className="font-mono text-xs">{result.request_id}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Sentence-level Analysis */}
      {result.sentences && result.sentences.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Sentence-by-Sentence Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
              {result.sentences.map((sentence: SentenceHighlight, index: number) => (
                <SentenceItem key={index} sentence={sentence} index={index} />
              ))}
            </div>
            <div className="mt-4 pt-4 border-t flex items-center gap-4 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-green-500/30"></span>
                Human-like
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-yellow-500/30"></span>
                Uncertain
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-red-500/30"></span>
                AI-like
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Plagiarism Matches */}
      {result.matches && result.matches.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-red-500" />
              Plagiarism Sources Found
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {result.matches.map((match, index) => (
                <div key={index} className="p-3 rounded-lg bg-red-500/5 border border-red-500/20">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="font-medium text-sm">{match.source_title}</p>
                      {match.source_url && (
                        <p className="text-xs text-muted-foreground truncate max-w-md">
                          {match.source_url}
                        </p>
                      )}
                    </div>
                    <Badge variant="outline" className="bg-red-500/10 text-red-600 border-red-500/20 shrink-0">
                      {Math.round(match.similarity * 100)}% match
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function SentenceItem({ sentence, index }: { sentence: SentenceHighlight; index: number }) {
  const aiProb = Math.round(sentence.ai_probability * 100)
  
  const getBgColor = (prob: number) => {
    if (prob >= 70) return 'bg-red-500/10 border-red-500/20'
    if (prob >= 50) return 'bg-yellow-500/10 border-yellow-500/20'
    return 'bg-green-500/10 border-green-500/20'
  }
  
  const getTextColor = (prob: number) => {
    if (prob >= 70) return 'text-red-600'
    if (prob >= 50) return 'text-yellow-600'
    return 'text-green-600'
  }

  return (
    <div className={`p-3 rounded-lg border ${getBgColor(aiProb)} transition-colors`}>
      <div className="flex items-start gap-3">
        <span className="text-xs text-muted-foreground font-mono shrink-0 mt-0.5">
          {String(index + 1).padStart(2, '0')}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm leading-relaxed">{sentence.sentence}</p>
          <div className="mt-2 flex items-center gap-2">
            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
              <div 
                className={`h-full transition-all ${aiProb >= 70 ? 'bg-red-500' : aiProb >= 50 ? 'bg-yellow-500' : 'bg-green-500'}`}
                style={{ width: `${aiProb}%` }}
              />
            </div>
            <span className={`text-xs font-medium ${getTextColor(aiProb)}`}>
              {aiProb}% AI
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
