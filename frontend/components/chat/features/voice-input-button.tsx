import { IconButton } from "@langchain/macaw-components/IconButton"
import { MicrophoneIcon } from "@phosphor-icons/react/dist/ssr/Microphone"
import { StopIcon } from "@phosphor-icons/react/dist/ssr/Stop"

interface VoiceInputButtonProps {
  isListening: boolean
  disabled?: boolean
  onClick: () => void
}

export function VoiceInputButton({ isListening, disabled, onClick }: VoiceInputButtonProps) {
  return (
    <IconButton
      type="button"
      icon={isListening ? StopIcon : MicrophoneIcon}
      iconWeight={isListening ? "fill" : "regular"}
      label={isListening ? "Stop voice input" : "Start voice input"}
      aria-pressed={isListening}
      color={isListening ? "primary" : "secondary"}
      variant="plain"
      size="sm"
      disabled={disabled}
      onClick={onClick}
    />
  )
}
