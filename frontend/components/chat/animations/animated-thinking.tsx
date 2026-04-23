"use client"

import React, { useEffect, useState, useMemo } from "react"

const thinkingWords = [
  "Thinking...",
  "Noodling...",
  "Percolating...",
  "Marinating...",
  "Brewing...",
  "Tinkering...",
  "Skedaddling...",
  "Sashaying...",
  "Boogying...",
  "Bopping...",
  "Getting Jiggy...",
  "Frolicking...",
  "Cooking...",
  "Doodling...",
  "Pondering...",
  "Conjuring...",
  "Gallivanting...",
  "Grooving...",
]

// Fisher-Yates shuffle algorithm
function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }
  return shuffled
}

export function AnimatedThinking() {
  // Create a shuffled version of the words once on mount
  const shuffledWords = useMemo(() => shuffleArray(thinkingWords), [])

  // Start at index 0 of the shuffled array
  const [wordIndex, setWordIndex] = useState(0)

  useEffect(() => {
    const getRandomInterval = () => {
      // Random interval between 2.5-4.5 seconds (slower)
      return 2500 + Math.random() * 2000
    }

    const scheduleNext = () => {
      const timeout = setTimeout(() => {
        // Move to next word in shuffled array, loop back to start when done
        setWordIndex((prev) => (prev + 1) % shuffledWords.length)
        scheduleNext()
      }, getRandomInterval())

      return timeout
    }

    const timeout = scheduleNext()
    return () => clearTimeout(timeout)
  }, [shuffledWords.length])

  return (
    <span className="font-medium inline-flex items-center gap-0 relative">
      <span className="thinking-text-base">{shuffledWords[wordIndex].replace('...', '')}</span>
      <span className="inline-flex thinking-text-base">
        <span className="animate-bounce-dot" style={{ animationDelay: '0ms' }}>.</span>
        <span className="animate-bounce-dot" style={{ animationDelay: '150ms' }}>.</span>
        <span className="animate-bounce-dot" style={{ animationDelay: '300ms' }}>.</span>
      </span>
      <span className="thinking-gradient-overlay" aria-hidden="true">
        <span>{shuffledWords[wordIndex].replace('...', '')}</span>
        <span className="inline-flex">
          <span className="animate-bounce-dot" style={{ animationDelay: '0ms' }}>.</span>
          <span className="animate-bounce-dot" style={{ animationDelay: '150ms' }}>.</span>
          <span className="animate-bounce-dot" style={{ animationDelay: '300ms' }}>.</span>
        </span>
      </span>
    </span>
  )
}
