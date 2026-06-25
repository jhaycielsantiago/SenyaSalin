/**
 * useGestureRecognizer — React hook for webcam gesture recognition.
 *
 * Captures frames from the webcam at a configurable interval,
 * sends them to the FastAPI backend for classification, and
 * returns the latest recognition result.
 *
 * Temporal smoothing: the hook votes across the last N frames
 * before committing to a result, reducing flicker from single-frame misclassifications.
 */

import { useRef, useState, useCallback, useEffect } from 'react'
import { recognizeFrame } from '../api'
import type { RecognizeFrameResponse } from '../types'

interface UseGestureRecognizerOptions {
  intervalMs?: number       // How often to send a frame (default: 200ms = 5 fps)
  smoothingFrames?: number  // Majority vote window (default: 5 frames)
  enabled?: boolean
}

interface GestureRecognizerState {
  result: RecognizeFrameResponse | null
  isRunning: boolean
  error: string | null
}

export function useGestureRecognizer(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  options: UseGestureRecognizerOptions = {},
) {
  const { intervalMs = 200, smoothingFrames = 5, enabled = true } = options

  const [state, setState] = useState<GestureRecognizerState>({
    result: null,
    isRunning: false,
    error: null,
  })

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const recentLabels = useRef<(string | null)[]>([])

  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return null

    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0)

    // Return base64 JPEG (strip the data URL prefix)
    return canvas.toDataURL('image/jpeg', 0.8).split(',')[1]
  }, [videoRef, canvasRef])

  const getMajorityLabel = (labels: (string | null)[]): string | null => {
    const counts: Record<string, number> = {}
    for (const l of labels) {
      if (l) counts[l] = (counts[l] ?? 0) + 1
    }
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1])
    return sorted[0]?.[0] ?? null
  }

  const start = useCallback(() => {
    if (intervalRef.current) return
    setState(s => ({ ...s, isRunning: true, error: null }))

    intervalRef.current = setInterval(async () => {
      const b64 = captureFrame()
      if (!b64) return

      try {
        const result = await recognizeFrame(b64)

        // Smoothing: only surface the result when majority vote agrees
        recentLabels.current = [...recentLabels.current.slice(-(smoothingFrames - 1)), result.label]
        const majorityLabel = getMajorityLabel(recentLabels.current)

        if (majorityLabel === result.label) {
          setState(s => ({ ...s, result, error: null }))
        }
      } catch (err) {
        setState(s => ({ ...s, error: String(err) }))
      }
    }, intervalMs)
  }, [captureFrame, intervalMs, smoothingFrames])

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setState(s => ({ ...s, isRunning: false }))
    recentLabels.current = []
  }, [])

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])

  return { ...state, start, stop }
}
