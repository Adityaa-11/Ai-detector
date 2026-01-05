import { NextRequest, NextResponse } from 'next/server'

// Backend API URL - configurable via environment variable
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    // Validate input
    if (!body.text || typeof body.text !== 'string') {
      return NextResponse.json(
        { error: 'Text is required' },
        { status: 400 }
      )
    }

    const wordCount = body.text.trim().split(/\s+/).filter(Boolean).length
    if (wordCount < 50) {
      return NextResponse.json(
        { 
          error: `Text too short. Need at least 50 words, got ${wordCount}. Longer text = more accurate results.` 
        },
        { status: 400 }
      )
    }

    // Call the Python backend
    const response = await fetch(`${BACKEND_URL}/api/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: body.text,
        check_ai: body.check_ai ?? true,
        check_plagiarism: body.check_plagiarism ?? false,
        include_highlights: body.include_highlights ?? true,
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      return NextResponse.json(
        { error: errorData.detail || 'Detection failed' },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
    
  } catch (error) {
    console.error('Detection API error:', error)
    
    // Check if backend is unreachable
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { 
          error: 'Backend service unavailable. Please ensure the Python backend is running on port 8000.',
          hint: 'Run: cd backend && uvicorn api:app --reload'
        },
        { status: 503 }
      )
    }
    
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function GET() {
  try {
    // Health check proxy
    const response = await fetch(`${BACKEND_URL}/api/health`)
    
    if (!response.ok) {
      return NextResponse.json(
        { status: 'backend_error', backend_url: BACKEND_URL },
        { status: response.status }
      )
    }
    
    const data = await response.json()
    return NextResponse.json(data)
    
  } catch {
    return NextResponse.json(
      { 
        status: 'backend_unavailable',
        backend_url: BACKEND_URL,
        message: 'Python backend is not running'
      },
      { status: 503 }
    )
  }
}

