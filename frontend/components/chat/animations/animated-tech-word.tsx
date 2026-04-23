"use client"

import React, { useEffect, useState } from "react"

const sentences = [
  "What needs building?",
  "Drop the prompt",
  "How can I help?",
  "Ready when you are...",
  "What's the vision?",
  "Let's ship it!",
  "What's on your mind?",
  "Your move...",
  "Type away",
  "Just say the word...",
  "Let's get to work!",
  "Hit me with it!"
]

function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j: number = Math.floor(Math.random() * (i + 1))
    ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }
  return shuffled
}

interface AnimatedTechWordProps {
  disabled?: boolean
}

export function AnimatedTechWord({ disabled = false }: AnimatedTechWordProps) {
  const [shuffledSentences, setShuffledSentences] = useState(() => shuffleArray(sentences))
  const [sentenceIndex, setSentenceIndex] = useState(0)
  const [displayText, setDisplayText] = useState("")
  const [isDeleting, setIsDeleting] = useState(false)
  const [isPaused, setIsPaused] = useState(false)

  useEffect(() => {
    // Don't run animation if disabled
    if (disabled || isPaused) return

    const currentSentence = shuffledSentences[sentenceIndex]

    const typeSpeed = isDeleting ? 50 : 100
    const pauseAfterTyping = 2000
    const pauseAfterDeleting = 500

    const timeout = setTimeout(() => {
      if (!isDeleting) {
        if (displayText.length < currentSentence.length) {
          setDisplayText(currentSentence.substring(0, displayText.length + 1))
        } else {
          setIsPaused(true)
          setTimeout(() => {
            setIsPaused(false)
            setIsDeleting(true)
          }, pauseAfterTyping)
        }
      } else {
        if (displayText.length > 0) {
          setDisplayText(displayText.substring(0, displayText.length - 1))
        } else {
          setIsPaused(true)
          setIsDeleting(false)
          setDisplayText("")
          setTimeout(() => {
            setSentenceIndex((prev) => {
              const nextIndex = prev + 1
              if (nextIndex >= shuffledSentences.length) {
                setShuffledSentences(shuffleArray(sentences))
                return 0
              }
              return nextIndex
            })
            setIsPaused(false)
          }, pauseAfterDeleting)
        }
      }
    }, typeSpeed)

    return () => clearTimeout(timeout)
  }, [displayText, isDeleting, sentenceIndex, shuffledSentences, isPaused, disabled])

  // When disabled, show empty text
  if (disabled) {
    return (
      <span className="inline-block whitespace-nowrap">
        <span className="tech-word-text">&nbsp;</span>
      </span>
    )
  }

  return (
    <span className="inline-block whitespace-nowrap">
      <span className="tech-word-text">{displayText}</span>
      <span className={`tech-word-cursor ${isPaused ? 'blinking' : 'solid'}`}>&nbsp;</span>
    </span>
  )
}

