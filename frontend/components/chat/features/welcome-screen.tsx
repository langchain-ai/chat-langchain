"use client"

import { Button } from "@langchain/macaw-components/Button"
import { Logo } from "@langchain/macaw-components/Logo"
import { Text } from "@langchain/macaw-components/Text"
import { MessageComposer, type MessageComposerProps } from "../message-composer"

const suggestions = [
  "How do I build an agent with LangChain?",
  "How do I trace my app with LangSmith?",
  "How do I add human-in-the-loop approval?",
]

/** Welcome and active conversations share one composer and one application draft. */
export function WelcomeScreen(props: MessageComposerProps) {
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center overflow-y-auto px-4 py-8 sm:px-6">
      <div className="w-full max-w-3xl space-y-6">
        <div className="flex flex-col items-center gap-4 text-center">
          <Logo brand="langchain" variant="logomark" size="xl" />
          <div className="space-y-2">
            <Text as="h1" variant="h1">
              What can I help with?
            </Text>
            <Text color="secondary">
              Build with LangChain, LangGraph, and LangSmith.
            </Text>
          </div>
        </div>
        <MessageComposer {...props} />
        <div
          role="group"
          className="flex flex-wrap justify-center gap-2"
          aria-label="Suggested questions"
        >
          {suggestions.map((suggestion) => (
            <Button
              key={suggestion}
              variant="outlined"
              color="secondary"
              size="sm"
              disabled={!props.userId || props.isLoading}
              onClick={() => {
                props.onInputChange(suggestion)
                props.textareaRef?.current?.focus()
              }}
            >
              {suggestion}
            </Button>
          ))}
        </div>
      </div>
    </div>
  )
}
