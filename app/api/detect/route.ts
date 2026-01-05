import { NextRequest, NextResponse } from 'next/server'

// Backend API URL - configurable via environment variable
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  console.log('[API] BACKEND_URL:', BACKEND_URL)
  
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

    const targetUrl = `${BACKEND_URL}/api/detect`
    console.log('[API] Calling backend:', targetUrl)

    // Call the Python backend
    const response = await fetch(targetUrl, {
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

    console.log('[API] Backend response status:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('[API] Backend error:', errorText)
      return NextResponse.json(
        { error: `Backend error: ${response.status}`, details: errorText },
        { status: response.status }
      )
    }

    const data = await response.json()
    console.log('[API] Success, returning data')
    return NextResponse.json(data)
    
  } catch (error: any) {
    console.error('[API] Error:', error.message, error.stack)
    
    return NextResponse.json(
      { 
        error: 'Failed to connect to backend',
        details: error.message,
        backend_url: BACKEND_URL
      },
      { status: 503 }
    )
  }
}

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    backend_url: BACKEND_URL,
    message: 'Use POST to detect AI content'
  })
}
