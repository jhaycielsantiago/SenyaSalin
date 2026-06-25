// API response types — keep in sync with app/backend/schemas.py

export interface RecognizeFrameResponse {
  label: string | null
  intent: string | null
  phrase: string | null
  confidence: number
  tts_lang_code: string
  below_threshold: boolean
}

export interface GestureInfo {
  label: string
  intent: string
  description: string
  scenario_tags: string[]
}

export interface GestureListResponse {
  total: number
  gestures: GestureInfo[]
}

export interface BenchmarkResultSummary {
  model_name: string
  accuracy: number
  f1_weighted: number
  inference_time_ms_mean: number
}

export interface BenchmarkResultsResponse {
  results: BenchmarkResultSummary[]
}

export interface HealthResponse {
  status: string
  model_loaded: boolean
  model_name: string | null
  num_classes: number
  version: string
}
