"use client"

import { useEffect, useState } from "react"
import { LogOut } from "lucide-react"
import Image from "next/image"
import { useSearchParams } from "next/navigation"
import { AgentSettings, type AgentConfig } from "./agent-settings"
import { AuthModal } from "@/components/auth/AuthModal"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useAuth } from "@/lib/auth"

const AUTH_MODAL_DISMISSED_KEY = "chat-langchain-auth-modal-dismissed"

interface HeaderProps {
  showToolCalls?: boolean
  onToggleToolCalls?: () => void
  onNewChat?: () => void
  agentConfig?: AgentConfig
  onAgentConfigChange?: (config: AgentConfig) => void
  onShowShortcuts?: () => void
  forceShowTooltip?: number
  showSettingsDialog?: boolean
  onSettingsDialogChange?: (open: boolean) => void
}

export function Header({ showToolCalls = false, onToggleToolCalls, onNewChat, agentConfig, onAgentConfigChange, onShowShortcuts, forceShowTooltip, showSettingsDialog, onSettingsDialogChange }: HeaderProps) {
  const [showAuthModal, setShowAuthModal] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)
  const { user, loading, signOut } = useAuth()
  const searchParams = useSearchParams()

  useEffect(() => {
    if (!loading && !user) {
      if (sessionStorage.getItem(AUTH_MODAL_DISMISSED_KEY) === "true") {
        return
      }
      setShowAuthModal(true)
    }
  }, [loading, user])

  useEffect(() => {
    const error = searchParams.get("auth_error")
    if (!error) return

    const description = searchParams.get("auth_error_description")
    const lastProvider =
      typeof window === "undefined" ? null : localStorage.getItem("lastAuthProvider")
    const githubConflictMessage =
      "We couldn't sign you in with GitHub. This can happen when the same GitHub account is connected to multiple LangSmith accounts with different emails. Please sign in with email/password or another provider."

    setAuthError(
      lastProvider === "github"
        ? githubConflictMessage
        : description || "Sign in failed. Please try another sign-in method."
    )
    sessionStorage.removeItem(AUTH_MODAL_DISMISSED_KEY)
    setShowAuthModal(true)
  }, [searchParams])

  const handleAuthModalOpenChange = (open: boolean) => {
    setShowAuthModal(open)
    if (!open) {
      sessionStorage.setItem(AUTH_MODAL_DISMISSED_KEY, "true")
      setAuthError(null)
    }
  }

  const handleSignOut = () => {
    sessionStorage.setItem(AUTH_MODAL_DISMISSED_KEY, "true")
    void signOut()
  }

  return (
    <>
      <header className="border-b border-border/60 bg-background h-16 flex items-center">
        <div className="flex items-center justify-between w-full px-4 sm:px-6">
          <div className="flex items-center">
            <Image
              src="/assets/images/ChatLangChain-logo.svg"
              alt="Chat LangChain"
              width={200}
              height={32}
              className="object-contain"
              style={{ width: "200px", height: "32px" }}
              priority
            />
          </div>

          <div className="flex items-center gap-3">
            {agentConfig && onAgentConfigChange && (
              <AgentSettings
                config={agentConfig}
                onConfigChange={onAgentConfigChange}
                onShowShortcuts={onShowShortcuts}
                forceShowTooltip={forceShowTooltip}
                open={showSettingsDialog}
                onOpenChange={onSettingsDialogChange}
              />
            )}
            {!loading && !user && (
              <button
                type="button"
                onClick={() => setShowAuthModal(true)}
                className="hidden sm:inline-flex items-center gap-2 px-3 py-2 border border-primary/20 hover:border-primary/40 rounded-full text-sm font-medium text-foreground/70 hover:text-foreground transition-all duration-200"
              >
                Sign in
              </button>
            )}
            {!loading && user && (
              <UserAccountMenu
                name={user.user_metadata?.full_name || user.email || "User"}
                email={user.email || ""}
                image={user.user_metadata?.avatar_url}
                onSignOut={handleSignOut}
              />
            )}
            <button
              onClick={onNewChat}
              className="group inline-flex items-center gap-2 px-3 sm:px-4 py-2 bg-gradient-to-r from-primary/10 to-primary/5 hover:from-primary/20 hover:to-primary/10 border border-primary/20 hover:border-primary/40 rounded-full text-sm font-medium text-foreground/80 hover:text-foreground transition-all duration-200 hover:scale-105 hover:shadow-lg"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-primary group-hover:rotate-12 transition-transform duration-200"
              >
                <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
              </svg>
              <span className="hidden sm:inline">New Chat</span>
            </button>
          </div>
        </div>
      </header>
      <AuthModal
        open={showAuthModal}
        onOpenChange={handleAuthModalOpenChange}
        initialError={authError}
      />
    </>
  )
}

function UserAccountMenu({
  name,
  email,
  image,
  onSignOut,
}: {
  name: string
  email: string
  image?: string
  onSignOut: () => void
}) {
  const initials = name
    .split(" ")
    .map((part) => part[0])
    .join("")
    .toUpperCase()
    .slice(0, 2) || "U"

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="relative h-9 w-9 rounded-full">
          <Avatar className="h-9 w-9">
            <AvatarImage src={image} alt={name} />
            <AvatarFallback>{initials}</AvatarFallback>
          </Avatar>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-56" align="end" forceMount>
        <DropdownMenuLabel className="font-normal">
          <div className="flex flex-col space-y-1">
            <p className="text-sm font-medium leading-none">{name}</p>
            <p className="text-xs leading-none text-muted-foreground">
              {email}
            </p>
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={onSignOut}>
          <LogOut className="mr-2 h-4 w-4" />
          <span>Sign out</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
