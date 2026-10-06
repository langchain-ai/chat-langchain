export interface KeyboardEventLike {
  key: string
  keyCode: number
  shiftKey: boolean
  nativeEvent: {
    isComposing?: boolean
  }
}

export function shouldSubmitOnEnter(event: KeyboardEventLike): boolean {
  if (event.nativeEvent.isComposing || event.keyCode === 229) return false
  return event.key === "Enter" && !event.shiftKey
}
