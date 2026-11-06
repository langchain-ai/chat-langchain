"use client"

import { useEffect, useState } from "react"
import {
  SignOutRegularIcon,
  SidebarSimpleRightRegularIcon,
  PlusIcon,
} from "@langchain/macaw-components/icons"
import Image from "next/image"
import { useSearchParams } from "next/navigation"
import { AuthModal } from "@/components/auth/AuthModal"
import { Avatar } from "@langchain/macaw-components/Avatar"
import { Button } from "@langchain/macaw-components/Button"
import { IconButton } from "@langchain/macaw-components/IconButton"
import { Text } from "@langchain/macaw-components/Text"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@langchain/macaw-components/DropdownMenu"
import { useAuth } from "@/lib/auth"

const AUTH_MODAL_DISMISSED_KEY = "chat-langchain-auth-modal-dismissed"

interface HeaderProps {
  onNewChat?: () => void
  onOpenSidebar?: () => void
  showInitialAuthPrompt?: boolean
}

export function Header({
  onNewChat,
  onOpenSidebar,
  showInitialAuthPrompt = true,
}: HeaderProps) {
  const [showAuthModal, setShowAuthModal] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)
  const { user, loading, signOut } = useAuth()
  const searchParams = useSearchParams()

  useEffect(() => {
    if (showInitialAuthPrompt && !loading && !user) {
      if (sessionStorage.getItem(AUTH_MODAL_DISMISSED_KEY) === "true") {
        return
      }
      setShowAuthModal(true)
    }
  }, [loading, user, showInitialAuthPrompt])

  useEffect(() => {
    const error = searchParams.get("auth_error")
    if (!error) return

    const lastProvider =
      typeof window === "undefined"
        ? null
        : localStorage.getItem("lastAuthProvider")
    const githubConflictMessage =
      "We couldn't sign you in with GitHub. This can happen when the same GitHub account is connected to multiple LangSmith accounts with different emails. Please sign in with email/password or another provider."

    setAuthError(
      lastProvider === "github"
        ? githubConflictMessage
        : "Sign in failed. Please try another sign-in method."
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
      <header className="flex h-12 shrink-0 items-center justify-between gap-3 bg-background px-3 sm:px-6">
        <div className="flex min-w-0 items-center gap-2">
          {onOpenSidebar && (
            <IconButton
              icon={SidebarSimpleRightRegularIcon}
              label="Open conversation history"
              variant="plain"
              onClick={onOpenSidebar}
              className="md:hidden"
            />
          )}
          <Image
            src="/assets/images/ChatLangChain-logo.svg"
            alt="Chat LangChain"
            width={200}
            height={32}
            className="h-auto w-36 sm:w-[200px]"
            priority
          />
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {!loading && !user && (
            <Button variant="outlined" onClick={() => setShowAuthModal(true)}>
              Sign in
            </Button>
          )}
          {!loading && user && (
            <UserAccountMenu
              name={user.user_metadata?.full_name || user.email || "User"}
              email={user.email || ""}
              image={user.user_metadata?.avatar_url}
              onSignOut={handleSignOut}
            />
          )}
          <Button
            leftDecorator={PlusIcon}
            onClick={onNewChat}
            className="hidden sm:inline-flex"
          >
            New chat
          </Button>
          <IconButton
            icon={PlusIcon}
            label="New chat"
            onClick={onNewChat}
            className="sm:hidden"
          />
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
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="plain" aria-label={`Account menu for ${name}`}>
          <Avatar label={name} imageUrl={image} size="md" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-64" align="end">
        <DropdownMenuLabel>
          <div className="flex min-w-0 flex-col gap-1">
            <Text variant="sm" weight="medium" className="truncate">
              {name}
            </Text>
            <Text variant="xs" color="tertiary" className="truncate">
              {email}
            </Text>
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={onSignOut}>
          <SignOutRegularIcon className="mr-2 size-4" aria-hidden="true" />
          Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
