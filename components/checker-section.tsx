"use client"

import { useState, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { 
  Upload, 
  FileText, 
  Sparkles, 
  Copy, 
  Download, 
  RefreshCw, 
  AlertCircle,
  CheckCircle2,
  Loader2,
  X
} from "lucide-react"
import { ResultsDisplay } from "@/components/results-display"
import type { DetectionResult } from "@/lib/types"

export function CheckerSection() {
  const [text, setText] = useState("")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [checkPlagiarism, setCheckPlagiarism] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [copied, setCopied] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const wordCount = text.trim().split(/\s+/).filter(Boolean).length
  const charCount = text.length
  const minWords = 50

  const handleAnalyze = async () => {
    if (wordCount < minWords) {
      setError(`Please enter at least ${minWords} words for accurate analysis. You have ${wordCount} words.`)
      return
    }

    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          check_ai: true,
          check_plagiarism: checkPlagiarism,
          include_highlights: true,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || data.detail || 'Analysis failed')
      }

      setResult(data)
    } catch (err) {
      console.error('Analysis error:', err)
      setError(err instanceof Error ? err.message : 'Failed to analyze text. Please try again.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleFileUpload = async (file: File) => {
    setSelectedFile(file)
    setError(null)
    setIsAnalyzing(true)
    setResult(null)

    try {
      // For text files, read and analyze directly
      if (file.name.toLowerCase().endsWith('.txt')) {
        const text = await file.text()
        setText(text)
        
        const response = await fetch('/api/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text,
            check_ai: true,
            check_plagiarism: checkPlagiarism,
            include_highlights: true,
          }),
        })

        const data = await response.json()
        if (!response.ok) {
          throw new Error(data.error || 'Analysis failed')
        }
        setResult(data)
      } else {
        // For PDF/DOCX, use file upload endpoint
        const formData = new FormData()
        formData.append('file', file)

        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        })

        const data = await response.json()
        if (!response.ok) {
          throw new Error(data.error || 'File processing failed')
        }
        setResult(data)
      }
    } catch (err) {
      console.error('Upload error:', err)
      setError(err instanceof Error ? err.message : 'Failed to process file')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    const file = e.dataTransfer.files[0]
    if (file) {
      const allowed = ['.txt', '.pdf', '.docx']
      const isAllowed = allowed.some(ext => file.name.toLowerCase().endsWith(ext))
      if (isAllowed) {
        handleFileUpload(file)
      } else {
        setError('Unsupported file type. Please use .txt, .pdf, or .docx files.')
      }
    }
  }, [checkPlagiarism])

  const handleClear = () => {
    setText("")
    setResult(null)
    setError(null)
    setSelectedFile(null)
  }

  const handleCopy = async () => {
    if (text) {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleDownloadReport = () => {
    if (!result) return

    const report = `
AI & Plagiarism Detection Report
================================
Generated: ${new Date().toLocaleString()}
Request ID: ${result.request_id}

SUMMARY
-------
Word Count: ${result.word_count}
Processing Time: ${result.processing_time}s

AI DETECTION
------------
AI Probability: ${Math.round((result.ai_probability ?? 0) * 100)}%
Human Probability: ${Math.round((result.human_probability ?? 0) * 100)}%
Verdict: ${result.verdict}
Confidence: ${result.confidence}
Model: ${result.model_used || 'RoBERTa Ensemble'}

${result.plagiarism_score !== undefined ? `
PLAGIARISM CHECK
----------------
Plagiarism Score: ${Math.round(result.plagiarism_score)}%
Is Plagiarized: ${result.is_plagiarized ? 'Yes' : 'No'}
Matches Found: ${result.matches?.length ?? 0}
` : ''}

${result.sentences ? `
SENTENCE ANALYSIS
-----------------
${result.sentences.map((s, i) => 
  `${i + 1}. [${Math.round(s.ai_probability * 100)}% AI] ${s.sentence}`
).join('\n')}
` : ''}

---
Analyzed text:
${text}
    `.trim()

    const blob = new Blob([report], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ai-detection-report-${result.request_id}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section id="checker" className="container mx-auto px-4 py-12">
      <Card className="max-w-5xl mx-auto border-border shadow-lg">
        <CardHeader>
          <CardTitle className="text-2xl">Check Your Content</CardTitle>
          <CardDescription>
            Paste your text or upload a file to detect AI-generated content and plagiarism
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Error Alert */}
          {error && (
            <Alert variant="destructive" className="mb-6">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="flex items-center justify-between">
                <span>{error}</span>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => setError(null)}
                  className="h-6 w-6 p-0"
                >
                  <X className="h-4 w-4" />
                </Button>
              </AlertDescription>
            </Alert>
          )}

          <Tabs defaultValue="text" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="text" className="gap-2">
                <FileText className="w-4 h-4" />
                Paste Text
              </TabsTrigger>
              <TabsTrigger value="upload" className="gap-2">
                <Upload className="w-4 h-4" />
                Upload File
              </TabsTrigger>
            </TabsList>

            <TabsContent value="text" className="space-y-4">
              <div className="relative">
                <Textarea
                  placeholder="Paste your text here to check for AI-generated content and plagiarism. For best results, use at least 50 words..."
                  className="min-h-[300px] resize-none text-base pr-24"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  disabled={isAnalyzing}
                />
                <div className="absolute bottom-3 right-3 text-xs text-muted-foreground bg-background px-2 py-1 rounded border">
                  <span className={wordCount < minWords ? 'text-destructive' : 'text-green-600'}>
                    {wordCount}
                  </span>
                  /{minWords} words • {charCount} chars
                </div>
              </div>

              {/* Options */}
              <div className="flex items-center gap-6 p-4 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-2">
                  <Switch 
                    id="plagiarism" 
                    checked={checkPlagiarism}
                    onCheckedChange={setCheckPlagiarism}
                    disabled={isAnalyzing}
                  />
                  <Label htmlFor="plagiarism" className="text-sm cursor-pointer">
                    Check for plagiarism
                  </Label>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap gap-3 items-center justify-between">
                <div className="flex gap-2">
                  <Button 
                    onClick={handleAnalyze} 
                    disabled={!text.trim() || isAnalyzing || wordCount < minWords} 
                    className="gap-2"
                    size="lg"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4" />
                        Analyze Content
                      </>
                    )}
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={handleClear} 
                    disabled={(!text && !result) || isAnalyzing}
                    className="gap-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Clear
                  </Button>
                </div>

                <div className="flex gap-2">
                  <Button 
                    variant="ghost" 
                    size="icon"
                    onClick={handleCopy}
                    disabled={!text.trim()}
                    title="Copy text"
                  >
                    {copied ? (
                      <CheckCircle2 className="w-4 h-4 text-green-500" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </Button>
                  <Button 
                    variant="ghost" 
                    size="icon"
                    onClick={handleDownloadReport}
                    disabled={!result}
                    title="Download report"
                  >
                    <Download className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="upload" className="space-y-4">
              <div 
                className={`
                  border-2 border-dashed rounded-lg p-12 text-center transition-all cursor-pointer
                  ${isDragging 
                    ? 'border-primary bg-primary/5' 
                    : 'border-border hover:border-primary/50'
                  }
                  ${isAnalyzing ? 'opacity-50 pointer-events-none' : ''}
                `}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.pdf,.docx"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleFileUpload(file)
                  }}
                />
                
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-12 h-12 mx-auto mb-4 text-primary animate-spin" />
                    <p className="text-sm text-muted-foreground mb-2">Processing file...</p>
                  </>
                ) : selectedFile ? (
                  <>
                    <FileText className="w-12 h-12 mx-auto mb-4 text-primary" />
                    <p className="text-sm font-medium mb-2">{selectedFile.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(selectedFile.size / 1024).toFixed(1)} KB
                    </p>
                  </>
                ) : (
                  <>
                    <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-sm text-muted-foreground mb-2">
                      Drag and drop your file here, or click to browse
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Supports: PDF, DOCX, TXT (Max 10MB)
                    </p>
                  </>
                )}
                
                <Button 
                  className="mt-4" 
                  variant="outline"
                  disabled={isAnalyzing}
                  onClick={(e) => {
                    e.stopPropagation()
                    fileInputRef.current?.click()
                  }}
                >
                  {selectedFile ? 'Choose Another File' : 'Select File'}
                </Button>
              </div>

              {/* Options for file upload */}
              <div className="flex items-center gap-6 p-4 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-2">
                  <Switch 
                    id="plagiarism-upload" 
                    checked={checkPlagiarism}
                    onCheckedChange={setCheckPlagiarism}
                    disabled={isAnalyzing}
                  />
                  <Label htmlFor="plagiarism-upload" className="text-sm cursor-pointer">
                    Check for plagiarism
                  </Label>
                </div>
              </div>
            </TabsContent>
          </Tabs>

          {/* Results */}
          {result && <ResultsDisplay result={result} />}
        </CardContent>
      </Card>
    </section>
  )
}
