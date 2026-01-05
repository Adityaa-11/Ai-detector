import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File | null
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      )
    }

    // Check file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      return NextResponse.json(
        { error: 'File too large. Maximum size is 10MB.' },
        { status: 400 }
      )
    }

    // Check file type
    const allowedTypes = ['.txt', '.pdf', '.docx']
    const fileName = file.name.toLowerCase()
    const isAllowed = allowedTypes.some(ext => fileName.endsWith(ext))
    
    if (!isAllowed) {
      return NextResponse.json(
        { error: 'Unsupported file type. Use .txt, .pdf, or .docx' },
        { status: 400 }
      )
    }

    // For now, handle .txt files directly in Next.js
    // PDF and DOCX require the Python backend
    if (fileName.endsWith('.txt')) {
      const text = await file.text()
      const wordCount = text.trim().split(/\s+/).filter(Boolean).length
      
      if (wordCount < 50) {
        return NextResponse.json(
          { error: `File contains too little text (${wordCount} words). Need at least 50 words.` },
          { status: 400 }
        )
      }

      // Call detection API
      const response = await fetch(`${BACKEND_URL}/api/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          check_ai: true,
          check_plagiarism: false,
          include_highlights: true,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        return NextResponse.json(
          { error: errorData.detail || 'Detection failed' },
          { status: response.status }
        )
      }

      return NextResponse.json(await response.json())
    }

    // For PDF/DOCX, forward to Python backend
    const backendFormData = new FormData()
    backendFormData.append('file', file)

    const response = await fetch(`${BACKEND_URL}/api/upload`, {
      method: 'POST',
      body: backendFormData,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      return NextResponse.json(
        { error: errorData.detail || 'File processing failed' },
        { status: response.status }
      )
    }

    return NextResponse.json(await response.json())

  } catch (error) {
    console.error('Upload API error:', error)
    
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { error: 'Backend service unavailable' },
        { status: 503 }
      )
    }
    
    return NextResponse.json(
      { error: 'Failed to process file' },
      { status: 500 }
    )
  }
}

