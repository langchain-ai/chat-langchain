import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-chat-ring focus-visible:ring-chat-ring/50 focus-visible:ring-[3px] aria-invalid:ring-chat-destructive/20 dark:aria-invalid:ring-chat-destructive/40 aria-invalid:border-chat-destructive",
  {
    variants: {
      variant: {
        default: "bg-chat-primary text-chat-primary-foreground hover:bg-chat-primary/90",
        destructive:
          "bg-chat-destructive text-white hover:bg-chat-destructive/90 focus-visible:ring-chat-destructive/20 dark:focus-visible:ring-chat-destructive/40 dark:bg-chat-destructive/60",
        outline:
          "border bg-chat-background shadow-xs hover:bg-chat-accent hover:text-chat-accent-foreground dark:bg-chat-input/30 dark:border-chat-input dark:hover:bg-chat-input/50",
        secondary:
          "bg-chat-secondary text-chat-secondary-foreground hover:bg-chat-secondary/80",
        ghost:
          "hover:bg-chat-accent hover:text-chat-accent-foreground dark:hover:bg-chat-accent/50",
        link: "text-chat-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2 has-[>svg]:px-3",
        sm: "h-8 rounded-md gap-1.5 px-3 has-[>svg]:px-2.5",
        lg: "h-10 rounded-md px-6 has-[>svg]:px-4",
        icon: "size-9",
        "icon-sm": "size-8",
        "icon-lg": "size-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : "button"

  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
