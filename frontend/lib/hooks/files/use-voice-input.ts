/**
 * Voice Input Hook
 *
 * Provides speech-to-text functionality using the Web Speech API.
 * Streams recognized speech in real-time to the provided callback.
 */

import { useState, useCallback, useRef, useEffect } from "react"

// ============================================================================
// Types
// ============================================================================

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList
  resultIndex: number
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string
  message?: string
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean
  interimResults: boolean
  lang: string
  start: () => void
  stop: () => void
  abort: () => void
  onresult: ((event: SpeechRecognitionEvent) => void) | null
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null
  onend: (() => void) | null
  onstart: (() => void) | null
}

declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognition
    webkitSpeechRecognition: new () => SpeechRecognition
  }
}

export interface UseVoiceInputReturn {
  isListening: boolean
  isSupported: boolean
  error: string | null
  interimTranscript: string
  startListening: () => void
  stopListening: () => void
  toggleListening: () => void
}

// ============================================================================
// Hook
// ============================================================================

/**
 * Hook for voice-to-text input using the Web Speech API.
 * Provides real-time interim results and final transcripts.
 *
 * @param onTranscript - Called with finalized transcript text
 * @returns Voice input state and controls
 *
 * @example
 * ```tsx
 * const { isListening, interimTranscript, toggleListening } = useVoiceInput({
 *   onTranscript: (text) => appendToInput(text),
 * })
 * ```
 */
export function useVoiceInput({
  onTranscript,
}: {
  onTranscript: (text: string) => void
}): UseVoiceInputReturn {
  const [isListening, setIsListening] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isSupported, setIsSupported] = useState(false)
  const [interimTranscript, setInterimTranscript] = useState("")

  const recognitionRef = useRef<SpeechRecognition | null>(null)
  const isStartingRef = useRef(false)

  // Use ref for callback to avoid recreating recognition on every render
  const onTranscriptRef = useRef(onTranscript)
  useEffect(() => {
    onTranscriptRef.current = onTranscript
  }, [onTranscript])

  // Check for browser support on mount
  useEffect(() => {
    const SpeechRecognitionAPI =
      typeof window !== "undefined"
        ? window.SpeechRecognition || window.webkitSpeechRecognition
        : null

    setIsSupported(!!SpeechRecognitionAPI)

    if (SpeechRecognitionAPI && !recognitionRef.current) {
      const recognition = new SpeechRecognitionAPI()
      // Use non-continuous mode for better compatibility
      // User can click again to continue recording
      recognition.continuous = false
      recognition.interimResults = true
      recognition.lang = "en-US"
      // @ts-expect-error - maxAlternatives exists but not in types
      recognition.maxAlternatives = 1

      recognition.onstart = () => {
        isStartingRef.current = false
        setIsListening(true)
        setError(null)
        setInterimTranscript("")
      }

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        let interim = ""
        let final = ""

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i]
          const transcript = result[0].transcript

          if (result.isFinal) {
            final += transcript
          } else {
            interim += transcript
          }
        }

        setInterimTranscript(interim)

        if (final) {
          onTranscriptRef.current(final)
          setInterimTranscript("")
        }
      }

      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        isStartingRef.current = false

        // "aborted" is expected when user stops listening or component unmounts
        if (event.error === "aborted") {
          setIsListening(false)
          return
        }

        console.error("Speech recognition error:", event.error)

        let errorMessage: string
        switch (event.error) {
          case "no-speech":
            errorMessage = "No speech detected. Please try again."
            break
          case "audio-capture":
            errorMessage = "No microphone found. Please check your microphone."
            break
          case "not-allowed":
            errorMessage = "Microphone access denied. Please allow microphone access."
            break
          case "network":
            errorMessage = "Speech recognition unavailable. Try Chrome or Edge, or check browser privacy settings."
            break
          default:
            errorMessage = `Error: ${event.error}`
        }

        setError(errorMessage)
        setIsListening(false)

        // Auto-dismiss error after 5 seconds
        setTimeout(() => setError(null), 5000)
      }

      recognition.onend = () => {
        isStartingRef.current = false
        setIsListening(false)
      }

      recognitionRef.current = recognition
    }

    // Cleanup on unmount
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort()
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only run once on mount - callbacks are handled via refs

  const startListening = useCallback(() => {
    // Prevent double-start with ref (more reliable than state for rapid clicks)
    if (recognitionRef.current && !isListening && !isStartingRef.current) {
      isStartingRef.current = true
      setError(null)
      try {
        recognitionRef.current.start()
      } catch (err) {
        // Recognition might already be running - silently ignore
        isStartingRef.current = false
      }
    }
  }, [isListening])

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      isStartingRef.current = false
      try {
        recognitionRef.current.stop()
      } catch {
        // Ignore errors when stopping
      }
    }
  }, [])

  const toggleListening = useCallback(() => {
    // Use both state and ref to determine current status
    if (isListening || isStartingRef.current) {
      stopListening()
    } else {
      startListening()
    }
  }, [isListening, startListening, stopListening])

  return {
    isListening,
    isSupported,
    error,
    interimTranscript,
    startListening,
    stopListening,
    toggleListening,
  }
}
