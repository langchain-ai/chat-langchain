import * as React from "react"

import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "file:text-chat-foreground placeholder:text-chat-muted-foreground selection:bg-chat-primary selection:text-chat-primary-foreground dark:bg-chat-input/30 border-chat-input h-9 w-full min-w-0 rounded-md border bg-transparent px-3 py-1 text-base shadow-xs transition-[color,box-shadow] outline-none file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
        "focus-visible:border-chat-ring focus-visible:ring-chat-ring/50 focus-visible:ring-[3px]",
        "aria-invalid:ring-chat-destructive/20 dark:aria-invalid:ring-chat-destructive/40 aria-invalid:border-chat-destructive",
        className
      )}
      {...props}
    />
  )
}

export { Input }
