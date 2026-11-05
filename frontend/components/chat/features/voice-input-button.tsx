import { Button } from "@langchain/macaw-components/Button"

interface VoiceInputButtonProps {
  isListening: boolean
  disabled?: boolean
  onClick: () => void
  size?: "sm" | "md"
}

export function VoiceInputButton({ isListening, disabled, onClick, size = "sm" }: VoiceInputButtonProps) {
  return (
    <Button
      type="button"
      aria-label={isListening ? "Stop listening" : "Voice input"}
      aria-pressed={isListening}
      color={isListening ? "primary" : "secondary"}
      variant={isListening ? "outlined" : "plain"}
      size={size}
      disabled={disabled}
      onClick={onClick}
    >
      {isListening ? "Stop listening" : "Voice"}
    </Button>
  )
}
