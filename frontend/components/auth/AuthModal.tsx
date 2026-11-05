"use client"

import { useEffect, useRef, useState } from "react"
import { Banner } from "@langchain/macaw-components/Banner"
import { Button } from "@langchain/macaw-components/Button"
import { Dialog, DialogContent } from "@langchain/macaw-components/Dialog"
import { Divider } from "@langchain/macaw-components/Divider"
import { Input } from "@langchain/macaw-components/Input"
import { Link } from "@langchain/macaw-components/Link"
import { Select } from "@langchain/macaw-components/Select"
import { Text } from "@langchain/macaw-components/Text"
import { trackEvent } from "@/components/providers/segment-provider"
import { useAuth, type OAuthProvider } from "@/lib/auth"
import type { AuthRegion } from "@/lib/auth/supabase"

interface AuthModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  initialError?: string | null
}

const LANGSMITH_REGION_URLS: Record<AuthRegion, string> = {
  us: "https://smith.langchain.com",
  eu: "https://eu.smith.langchain.com",
  apac: "https://apac.smith.langchain.com",
  aws: "https://aws.smith.langchain.com",
}

const REGION_OPTIONS: Record<AuthRegion, { label: string; location: string }> = {
  us: { label: "US", location: "Central" },
  eu: { label: "EU", location: "West" },
  apac: { label: "APAC", location: "Asia Pacific" },
  aws: { label: "AWS US", location: "East" },
}

const getLangSmithAuthUrl = (
  region: AuthRegion,
  mode: "sign_up" | "forgotten_password"
) => `${LANGSMITH_REGION_URLS[region]}/?mode=${mode}`

const SIGN_UP_CLICK_TRACKED_KEY = "chat-langchain-sign-up-click-tracked"

function RegionIcon({ region }: { region: AuthRegion }) {
  if (region === "us" || region === "aws") return <span className="text-sm">🇺🇸</span>
  if (region === "eu") return <span className="text-sm">🇪🇺</span>
  return <span className="text-sm">🇦🇺</span>
}

export function AuthModal({ open, onOpenChange, initialError }: AuthModalProps) {
  const {
    user,
    signIn,
    signInWithEmail,
    isConfigured,
    authRegion,
    availableAuthRegions,
    setAuthRegion,
  } = useAuth()
  const returnFocusRef = useRef<HTMLElement | null>(null)
  const isDiscordSupportedRegion = authRegion !== "apac" && authRegion !== "aws"
  const [loadingProvider, setLoadingProvider] =
    useState<OAuthProvider | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [lastUsedProvider, setLastUsedProvider] =
    useState<OAuthProvider | null>(null)
  const [isEmailLoading, setIsEmailLoading] = useState(false)

  useEffect(() => {
    if (typeof window === "undefined") return
    const lastUsed = localStorage.getItem("lastAuthProvider") as
      | OAuthProvider
      | null
    if (lastUsed) setLastUsedProvider(lastUsed)
  }, [])

  useEffect(() => {
    if (!open) {
      setLoadingProvider(null)
      setError(null)
      setEmail("")
      setPassword("")
    }
  }, [open])

  useEffect(() => {
    if (open && initialError) {
      setError(initialError)
    }
  }, [initialError, open])

  const handleOAuthSignIn = async (provider: OAuthProvider) => {
    try {
      setLoadingProvider(provider)
      setError(null)

      if (typeof window !== "undefined") {
        localStorage.setItem("lastAuthProvider", provider)
        setLastUsedProvider(provider)
      }

      await signIn(provider)
    } catch (error) {
      console.error("Sign in error:", error)
      setError(
        error instanceof Error
          ? error.message
          : "Failed to sign in. Please try again."
      )
      setLoadingProvider(null)
    }
  }

  const handleEmailAuth = async () => {
    try {
      setIsEmailLoading(true)
      setError(null)

      await signInWithEmail(email, password)
      onOpenChange(false)
    } catch (error) {
      console.error("Email auth error:", error)
      setError(
        error instanceof Error
          ? error.message
          : "Failed to authenticate. Please try again."
      )
    } finally {
      setIsEmailLoading(false)
    }
  }

  const handleSignUpClick = () => {
    if (user || typeof window === "undefined") return
    if (localStorage.getItem(SIGN_UP_CLICK_TRACKED_KEY)) return

    trackEvent("Auth Sign Up Clicked", { location: "auth_modal" })
    localStorage.setItem(SIGN_UP_CLICK_TRACKED_KEY, "true")
  }

  const isBusy = isEmailLoading || loadingProvider !== null
  const providerButtons = (
    <div className="flex items-stretch gap-2">
      {(["google", "github", ...(isDiscordSupportedRegion ? ["discord"] : [])] as OAuthProvider[]).map(provider => (
        <AuthProviderButton key={provider} provider={provider} loadingProvider={loadingProvider} lastUsedProvider={lastUsedProvider} disabled={!isConfigured || isBusy} onClick={handleOAuthSignIn} />
      ))}
    </div>
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title="Log in with your LangSmith account"
        description="Sign in to save your conversations, or continue as a guest."
        className="w-[calc(100vw-2rem)] max-w-lg"
        onOpenAutoFocus={() => { returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null }}
        onCloseAutoFocus={event => {
          const target = returnFocusRef.current
          if (target?.isConnected && target !== document.body) {
            event.preventDefault()
            target.focus()
          }
        }}
      >
        <div className="flex flex-col gap-5">
          <div className="flex flex-col gap-2">
            {/* Macaw Select names its trigger with aria-label; a <label> here would have no control. */}
            <Text as="span" variant="sm" weight="medium">Data region</Text>
            <Select<AuthRegion>
              aria-label="Data region"
              value={authRegion}
              onChange={region => { if (region) setAuthRegion(region) }}
              disabled={isBusy}
              options={(Object.keys(REGION_OPTIONS) as AuthRegion[]).map(region => ({
                value: region,
                label: REGION_OPTIONS[region].label,
                description: availableAuthRegions.includes(region) ? REGION_OPTIONS[region].location : "Not configured",
                disabled: !availableAuthRegions.includes(region),
              }))}
              renderValue={option => <span className="flex items-center gap-2"><RegionIcon region={option.value} />{option.label}</span>}
            />
            <Link href="https://docs.langchain.com/langsmith/regions-faq" target="_blank" rel="noopener noreferrer" variant="xs">Learn more about regions</Link>
          </div>
          {!isConfigured && <Banner intent="info">Login is not configured for this region. You can continue as a guest.</Banner>}
          {error && <div role="alert"><Banner intent="error">{error}</Banner></div>}
          {providerButtons}
          <div className="flex items-center gap-3">
            <Divider className="flex-1" />
            <Text variant="xs" color="tertiary">Or continue with email</Text>
            <Divider className="flex-1" />
          </div>
          <form className="flex flex-col gap-4" onSubmit={event => {
            event.preventDefault()
            if (isConfigured && email && password && !isBusy) void handleEmailAuth()
          }}>
            <Input id="auth-email" name="email" label="Email" type="email" autoComplete="username" placeholder="Your email address" value={email} onChange={setEmail} debounceMs={0} required disabled={!isConfigured || isBusy} />
            <Input id="auth-password" name="password" label="Password" type="password" autoComplete="current-password" placeholder="Your password" value={password} onChange={setPassword} debounceMs={0} required disabled={!isConfigured || isBusy} />
            <Link href={getLangSmithAuthUrl(authRegion, "forgotten_password")} target="_blank" rel="noreferrer" variant="xs">Forgot password?</Link>
            <Button type="submit" loading={isEmailLoading} disabled={!isConfigured || !email || !password || isBusy}>Continue</Button>
          </form>
          <div className="flex items-center justify-center gap-2">
            <Text variant="sm" color="secondary">Don't have an account?</Text>
            <Link href={getLangSmithAuthUrl(authRegion, "sign_up")} target="_blank" rel="noreferrer" variant="sm" onClick={handleSignUpClick}>Sign up</Link>
          </div>
          <Button variant="outlined" onClick={() => onOpenChange(false)}>Continue as guest</Button>
          <Text variant="xs" color="tertiary" className="text-center">
            By continuing, you agree to our <Link href="https://www.langchain.com/terms-of-service" target="_blank" rel="noreferrer" variant="xs">Terms of Service</Link> and <Link href="https://www.langchain.com/privacy-policy" target="_blank" rel="noreferrer" variant="xs">Privacy Policy</Link>.
          </Text>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function AuthProviderButton({
  provider,
  loadingProvider,
  lastUsedProvider,
  disabled,
  onClick,
}: {
  provider: OAuthProvider
  loadingProvider: OAuthProvider | null
  lastUsedProvider: OAuthProvider | null
  disabled: boolean
  onClick: (provider: OAuthProvider) => void
}) {
  const label = provider === "google" ? "Google" : provider === "github" ? "GitHub" : "Discord"
  return (
    <div className="flex min-w-0 flex-1 flex-col items-stretch gap-1">
      <Button variant="outlined" className="h-auto min-h-14" aria-label={`Continue with ${label}`} disabled={disabled} loading={loadingProvider === provider} onClick={() => onClick(provider)}>
        <span className="flex flex-col items-center gap-1"><ProviderIcon provider={provider} /><span>{label}</span></span>
      </Button>
      {lastUsedProvider === provider && <Text variant="xs" color="tertiary" className="text-center">Last used</Text>}
    </div>
  )
}

function ProviderIcon({ provider }: { provider: OAuthProvider }) {
  if (provider === "google") return <GoogleIcon />
  if (provider === "github") return <GitHubIcon />
  return <DiscordIcon />
}

function GoogleIcon() {
  return (
    <svg className="h-7 w-7" viewBox="0 0 33 33" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M30.5014 16.8109C30.5014 15.6598 30.4061 14.8198 30.1998 13.9487H16.7871V19.1442H24.6601C24.5014 20.4354 23.6442 22.3798 21.7394 23.6864L21.7127 23.8604L25.9536 27.08L26.2474 27.1087C28.9458 24.6665 30.5014 21.0731 30.5014 16.8109"
        fill="#4285F4"
      />
      <path
        d="M16.7853 30.4998C20.6424 30.4998 23.8804 29.2553 26.2456 27.1086L21.7377 23.6863C20.5313 24.5108 18.9123 25.0863 16.7853 25.0863C13.0076 25.0863 9.80128 22.6441 8.65832 19.2686L8.49078 19.2825L4.08111 22.627L4.02344 22.7841C6.37261 27.3574 11.198 30.4998 16.7853 30.4998Z"
        fill="#34A853"
      />
      <path
        d="M8.66062 19.269C8.35903 18.3979 8.1845 17.4645 8.1845 16.5001C8.1845 15.5356 8.35903 14.6023 8.64475 13.7312L8.63676 13.5456L4.17181 10.1475L4.02572 10.2156C3.05751 12.1134 2.50195 14.2445 2.50195 16.5001C2.50195 18.7556 3.05751 20.8867 4.02572 22.7845L8.66062 19.269"
        fill="#FBBC05"
      />
      <path
        d="M16.7854 7.9133C19.4679 7.9133 21.2774 9.04885 22.3092 9.9978L26.3409 6.14C23.8648 3.88445 20.6425 2.5 16.7854 2.5C11.198 2.5 6.37262 5.6422 4.02344 10.2155L8.64247 13.7311C9.80131 10.3555 13.0076 7.9133 16.7854 7.9133"
        fill="#EB4335"
      />
    </svg>
  )
}

function GitHubIcon() {
  return (
    <svg className="ml-[2px] h-7 w-7" viewBox="0 0 33 33" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M16.5 2.2207C14.5782 2.2207 12.6753 2.59922 10.8998 3.33465C9.1243 4.07008 7.51106 5.14801 6.15217 6.50691C3.40776 9.25131 1.86597 12.9735 1.86597 16.8547C1.86597 23.3229 6.06593 28.8107 11.8756 30.757C12.6073 30.8741 12.8415 30.4204 12.8415 30.0253V27.5522C8.78785 28.4302 7.92445 25.5912 7.92445 25.5912C7.25128 23.8937 6.30007 23.44 6.30007 23.44C4.96838 22.5327 6.40251 22.562 6.40251 22.562C7.86591 22.6644 8.64151 24.0693 8.64151 24.0693C9.91467 26.2936 12.0659 25.6351 12.9 25.2839C13.0317 24.3327 13.4122 23.6888 13.822 23.3229C10.5732 22.9571 7.16348 21.6986 7.16348 16.123C7.16348 14.4986 7.71957 13.1962 8.67078 12.1572C8.52444 11.7913 8.01225 10.2694 8.81712 8.29382C8.81712 8.29382 10.0464 7.8987 12.8415 9.78649C13.9976 9.46454 15.2561 9.30356 16.5 9.30356C17.7439 9.30356 19.0024 9.46454 20.1585 9.78649C22.9536 7.8987 24.1828 8.29382 24.1828 8.29382C24.9877 10.2694 24.4755 11.7913 24.3292 12.1572C25.2804 13.1962 25.8365 14.4986 25.8365 16.123C25.8365 21.7132 22.4121 22.9425 19.1487 23.3083C19.6756 23.762 20.1585 24.6546 20.1585 26.0156V30.0253C20.1585 30.4204 20.3926 30.8887 21.139 30.757C26.9487 28.7961 31.134 23.3229 31.134 16.8547C31.134 14.9329 30.7555 13.03 30.02 11.2545C29.2846 9.47904 28.2067 7.8658 26.8478 6.50691C25.4889 5.14801 23.8757 4.07008 22.1002 3.33465C20.3247 2.59922 18.4217 2.2207 16.5 2.2207Z"
        fill="currentColor"
      />
    </svg>
  )
}

function DiscordIcon() {
  return (
    <svg className="h-7 w-7" viewBox="0 0 33 33" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M26.7559 7.17821C24.8146 6.28944 22.7652 5.65891 20.66 5.30273C20.372 5.81771 20.1113 6.34755 19.8792 6.89004C17.6368 6.55213 15.3564 6.55213 13.114 6.89004C12.8817 6.3476 12.6211 5.81777 12.3331 5.30273C10.2266 5.66192 8.17582 6.29394 6.23256 7.18286C2.3747 12.8906 1.32889 18.4567 1.85179 23.9437C4.11106 25.6129 6.63981 26.8824 9.32815 27.6969C9.93348 26.8828 10.4691 26.0191 10.9294 25.115C10.0552 24.7885 9.2114 24.3856 8.40784 23.9111C8.61932 23.7577 8.82616 23.5997 9.02603 23.4463C11.3642 24.5459 13.9162 25.116 16.5001 25.116C19.0839 25.116 21.6359 24.5459 23.9741 23.4463C24.1763 23.6113 24.3831 23.7694 24.5923 23.9111C23.7872 24.3864 22.9418 24.79 22.0661 25.1173C22.5258 26.021 23.0615 26.884 23.6673 27.6969C26.3579 26.8857 28.8886 25.6168 31.1483 23.946C31.7619 17.5828 30.1002 12.0679 26.7559 7.17821ZM11.5917 20.5692C10.1346 20.5692 8.93074 19.2468 8.93074 17.62C8.93074 15.9932 10.0927 14.6592 11.5871 14.6592C13.0814 14.6592 14.276 15.9932 14.2504 17.62C14.2248 19.2468 13.0768 20.5692 11.5917 20.5692ZM21.4084 20.5692C19.9489 20.5692 18.7497 19.2468 18.7497 17.62C18.7497 15.9932 19.9117 14.6592 21.4084 14.6592C22.905 14.6592 24.0903 15.9932 24.0647 17.62C24.0392 19.2468 22.8934 20.5692 21.4084 20.5692Z"
        fill="#5865F2"
      />
    </svg>
  )
}
