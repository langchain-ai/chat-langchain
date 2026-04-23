"use client"

import { ChevronDown, Settings } from "lucide-react"
import Image from "next/image"
import { AgentSettings, type AgentConfig } from "./agent-settings"

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
  return (
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
  )
}
