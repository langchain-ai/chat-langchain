import { useEffect, useCallback } from 'react'

export type KeyboardShortcut = {
  key: string
  metaKey?: boolean
  ctrlKey?: boolean
  shiftKey?: boolean
  altKey?: boolean
  description: string
  category: string
}

export type KeyboardShortcutHandler = {
  handler: () => void
  shortcut: KeyboardShortcut
}

/**
 * Detects if the user is on macOS
 */
export const isMac = () => {
  if (typeof window === 'undefined') return false
  return navigator.userAgent.toUpperCase().indexOf('MAC') >= 0
}

/**
 * Gets the modifier key symbol for the current platform
 */
export const getModifierKey = () => {
  return isMac() ? '⌘' : 'Ctrl'
}

/**
 * Formats a keyboard shortcut for display
 */
export const formatShortcut = (shortcut: KeyboardShortcut): string => {
  const parts: string[] = []

  if (shortcut.metaKey || shortcut.ctrlKey) {
    parts.push(getModifierKey())
  }
  if (shortcut.shiftKey) {
    parts.push(isMac() ? '⇧' : 'Shift')
  }
  if (shortcut.altKey) {
    parts.push(isMac() ? '⌥' : 'Alt')
  }

  // Format the key
  let key = shortcut.key.toUpperCase()
  if (key === ' ') key = 'Space'
  if (key === 'ENTER') key = '↵'
  if (key === 'ESCAPE') key = 'Esc'
  if (key === 'BACKSPACE') key = '⌫'
  if (key === 'ARROWUP') key = '↑'
  if (key === 'ARROWDOWN') key = '↓'
  if (key === 'ARROWLEFT') key = '←'
  if (key === 'ARROWRIGHT') key = '→'

  parts.push(key)

  return parts.join(isMac() ? '' : '+')
}

/**
 * Checks if the event target is an input element where we should ignore shortcuts
 */
const isInputElement = (element: EventTarget | null): boolean => {
  if (!element || !(element instanceof HTMLElement)) return false

  const tagName = element.tagName.toLowerCase()
  const isContentEditable = element.isContentEditable

  return (
    tagName === 'input' ||
    tagName === 'textarea' ||
    tagName === 'select' ||
    isContentEditable
  )
}

/**
 * Checks if a keyboard event matches a shortcut definition
 */
const matchesShortcut = (
  event: KeyboardEvent,
  shortcut: KeyboardShortcut
): boolean => {
  const keyMatches = event.key.toLowerCase() === shortcut.key.toLowerCase()
  const metaMatches = shortcut.metaKey ? (event.metaKey || event.ctrlKey) : !event.metaKey && !event.ctrlKey
  const shiftMatches = shortcut.shiftKey ? event.shiftKey : !event.shiftKey
  const altMatches = shortcut.altKey ? event.altKey : !event.altKey

  return keyMatches && metaMatches && shiftMatches && altMatches
}

/**
 * Global keyboard shortcuts hook
 * Handles keyboard events and prevents conflicts with input fields
 */
export const useKeyboardShortcuts = (
  shortcuts: KeyboardShortcutHandler[],
  enabled: boolean = true
) => {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return

      // Allow shortcuts with Cmd/Ctrl modifier to work in input fields
      // Block shortcuts without modifiers when typing (except Escape)
      if (isInputElement(event.target)) {
        const hasModifier = event.metaKey || event.ctrlKey
        const isEscape = event.key.toLowerCase() === 'escape'
        if (!hasModifier && !isEscape) {
          return
        }
      }

      // Find matching shortcut
      for (const { shortcut, handler } of shortcuts) {
        if (matchesShortcut(event, shortcut)) {
          event.preventDefault()
          handler()
          break
        }
      }
    },
    [shortcuts, enabled]
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])
}

/**
 * Hook to get all available keyboard shortcuts grouped by category
 */
export const useAvailableShortcuts = (): Record<string, KeyboardShortcut[]> => {
  const shortcuts: KeyboardShortcut[] = [
    // Navigation
    {
      key: '/',
      metaKey: true,
      description: 'Toggle keyboard shortcuts',
      category: 'Navigation',
    },
    {
      key: 'b',
      metaKey: true,
      description: 'Toggle sidebar',
      category: 'Navigation',
    },
    {
      key: 'i',
      metaKey: true,
      description: 'Create new chat',
      category: 'Navigation',
    },
    {
      key: 's',
      metaKey: true,
      description: 'Toggle settings',
      category: 'Navigation',
    },

    // Model & Agent
    {
      key: 'j',
      metaKey: true,
      description: 'Cycle to next model',
      category: 'Model & Agent',
    },
    {
      key: 'k',
      metaKey: true,
      description: 'Cycle to next agent',
      category: 'Model & Agent',
    },
  ]

  // Group by category
  return shortcuts.reduce((acc, shortcut) => {
    if (!acc[shortcut.category]) {
      acc[shortcut.category] = []
    }
    acc[shortcut.category].push(shortcut)
    return acc
  }, {} as Record<string, KeyboardShortcut[]>)
}
