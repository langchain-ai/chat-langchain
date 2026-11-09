"use client"

import { useEffect, useState, type ComponentProps } from "react"
import { Avatar, type AvatarSize } from "@langchain/macaw-components/Avatar"
import { cn } from "@/lib/utils"

interface AvatarButtonProps
  extends Omit<ComponentProps<"button">, "children" | "color"> {
  /** Name used for the fallback initial and the avatar color. */
  label: string
  imageUrl?: string
  size?: AvatarSize
}

/**
 * A round, chromeless button around Macaw's Avatar, for account-menu triggers.
 *
 * Chrome treats the focus a menu returns to its trigger as `:focus-visible`,
 * even when the menu was opened and closed with a pointer. The ring stays
 * hidden after pointer use until the next key press, so it only shows for
 * keyboard users.
 */
export function AvatarButton({
  label,
  imageUrl,
  size = "md",
  className,
  type = "button",
  onPointerDown,
  ...props
}: AvatarButtonProps) {
  const [pointerActivated, setPointerActivated] = useState(false)

  useEffect(() => {
    if (!pointerActivated) return
    const reset = () => setPointerActivated(false)
    document.addEventListener("keydown", reset, true)
    return () => document.removeEventListener("keydown", reset, true)
  }, [pointerActivated])

  return (
    <button
      type={type}
      data-pointer-activated={pointerActivated || undefined}
      className={cn(
        "inline-flex shrink-0 rounded-full transition-opacity hover:opacity-80 focus-visible:outline-2 focus-visible:outline-offset-2 data-pointer-activated:outline-none",
        className
      )}
      onPointerDown={(event) => {
        setPointerActivated(true)
        onPointerDown?.(event)
      }}
      {...props}
    >
      <Avatar
        label={label}
        imageUrl={imageUrl}
        size={size}
        shape="circle"
        className="cursor-pointer"
      />
    </button>
  )
}
