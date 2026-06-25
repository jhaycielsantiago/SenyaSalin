// API client — all calls to the FastAPI backend go through here.

import type {
  RecognizeFrameResponse,
  GestureListResponse,
  BenchmarkResultsResponse,
  HealthResponse,
} from './types'

const BASE_URL = '/api'

export async function recognizeFrame(frameB64: string): Promise<RecognizeFrameResponse> {
  const res = await fetch(`${BASE_URL}/recognize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame_b64: frameB64 }),
  })
  if (!res.ok) throw new Error(`Recognize failed: ${res.status}`)
  return res.json()
}

export async function fetchGestures(): Promise<GestureListResponse> {
  const res = await fetch(`${BASE_URL}/gestures`)
  if (!res.ok) throw new Error(`Failed to fetch gestures: ${res.status}`)
  return res.json()
}

export async function fetchBenchmarkResults(): Promise<BenchmarkResultsResponse> {
  const res = await fetch(`${BASE_URL}/benchmark`)
  if (!res.ok) throw new Error(`Failed to fetch benchmark: ${res.status}`)
  return res.json()
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
  return res.json()
}
