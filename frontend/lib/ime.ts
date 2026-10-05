export function isImeComposing(
  event: { nativeEvent: { isComposing?: boolean }; keyCode?: number },
  isComposing: boolean,
) {
  return isComposing || event.nativeEvent.isComposing === true || event.keyCode === 229
}

export function shouldSubmitOnEnter(
  event: {
    key: string
    shiftKey: boolean
    nativeEvent: { isComposing?: boolean }
    keyCode?: number
  },
  isComposing: boolean,
) {
  return event.key === "Enter" && !event.shiftKey && !isImeComposing(event, isComposing)
}
