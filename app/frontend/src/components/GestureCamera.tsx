/**
 * GestureCamera — the live webcam recognition component.
 *
 * Renders the webcam feed, runs gesture recognition at 5 fps via the API,
 * and displays the recognized label, phrase, and confidence overlay.
 */

import { useRef, useEffect, useState } from 'react'
import { useGestureRecognizer } from '../hooks/useGestureRecognizer'
import type { RecognizeFrameResponse } from '../types'

export function GestureCamera() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [streamReady, setStreamReady] = useState(false)

  const { result, isRunning, error, start, stop } = useGestureRecognizer(videoRef, canvasRef)

  useEffect(() => {
    let stream: MediaStream | null = null

    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'user' } })
      .then((s) => {
        stream = s
        if (videoRef.current) {
          videoRef.current.srcObject = s
          videoRef.current.onloadedmetadata = () => setStreamReady(true)
        }
      })
      .catch(console.error)

    return () => {
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
      <div style={{ position: 'relative', width: 640, height: 480 }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 8 }}
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        {result && <ConfidenceOverlay result={result} />}
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={isRunning ? stop : start}
          disabled={!streamReady}
          style={{
            padding: '10px 24px',
            fontSize: 16,
            borderRadius: 6,
            cursor: streamReady ? 'pointer' : 'not-allowed',
            background: isRunning ? '#e53e3e' : '#38a169',
            color: '#fff',
            border: 'none',
          }}
        >
          {isRunning ? 'Stop Recognition' : 'Start Recognition'}
        </button>
      </div>

      {error && (
        <p style={{ color: '#e53e3e', fontSize: 14 }}>Error: {error}</p>
      )}
    </div>
  )
}

function ConfidenceOverlay({ result }: { result: RecognizeFrameResponse }) {
  const pct = Math.round(result.confidence * 100)

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 12,
        left: 12,
        right: 12,
        background: 'rgba(0,0,0,0.72)',
        borderRadius: 8,
        padding: '12px 16px',
        color: '#fff',
      }}
    >
      {result.below_threshold ? (
        <p style={{ margin: 0, fontSize: 14, color: '#fcd34d' }}>
          Hold your sign steady... ({pct}% confidence)
        </p>
      ) : (
        <>
          <p style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
            {result.label}
            <span style={{ fontSize: 14, fontWeight: 400, marginLeft: 8, color: '#a0aec0' }}>
              {pct}% confident
            </span>
          </p>
          {result.phrase && (
            <p style={{ margin: '6px 0 0', fontSize: 16, color: '#68d391' }}>
              {result.phrase}
            </p>
          )}
        </>
      )}
    </div>
  )
}
