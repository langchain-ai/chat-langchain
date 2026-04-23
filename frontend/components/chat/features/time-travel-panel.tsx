"use client"

import { History, GitBranch, Clock, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { Checkpoint } from "@/lib/hooks/threads"
import { formatDistanceToNow } from "date-fns"

interface TimeTravelPanelProps {
  checkpoints: Checkpoint[]
  currentCheckpointId?: string
  onJumpToCheckpoint: (checkpointId: string) => void
  onForkFromCheckpoint: (checkpointId: string) => void
  isOpen: boolean
  onClose: () => void
}

export function TimeTravelPanel({
  checkpoints,
  currentCheckpointId,
  onJumpToCheckpoint,
  onForkFromCheckpoint,
  isOpen,
  onClose,
}: TimeTravelPanelProps) {
  if (!isOpen) return null

  return (
    <div className="fixed right-0 top-0 h-full w-80 bg-card border-l border-border shadow-lg z-50 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <History className="w-5 h-5" />
            <h2 className="font-semibold">Time Travel</h2>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            X
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          {checkpoints.length} checkpoints in conversation
        </p>
      </div>

      {/* Checkpoint List */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-3">
          {checkpoints.map((checkpoint, idx) => {
            // Safely extract checkpoint ID with fallback
            const checkpointId = checkpoint.config?.configurable?.checkpoint_id ||
                                 checkpoint.metadata?.checkpoint_id ||
                                 `checkpoint-${idx}`
            const isCurrent = checkpointId === currentCheckpointId

            return (
              <div
                key={checkpointId}
                className={`p-3 rounded-lg border ${
                  isCurrent
                    ? "border-primary bg-primary/10"
                    : "border-border bg-muted/30 hover:bg-muted/50"
                } transition-colors cursor-pointer`}
                onClick={() => onJumpToCheckpoint(checkpointId)}
              >
                {/* Step Info */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Clock className="w-3 h-3 text-muted-foreground" />
                    <span className="text-xs font-medium">
                      Step {checkpoint.metadata?.step ?? idx}
                    </span>
                    {isCurrent && (
                      <span className="text-xs text-primary">● Current</span>
                    )}
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {checkpoint.created_at &&
                      formatDistanceToNow(new Date(checkpoint.created_at), {
                        addSuffix: true,
                      })}
                  </span>
                </div>

                {/* Metadata */}
                {checkpoint.metadata?.writes && (
                  <div className="text-xs text-muted-foreground mb-2">
                    {Object.keys(checkpoint.metadata.writes).map((key) => (
                      <div key={key} className="flex items-center gap-1">
                        <ChevronRight className="w-3 h-3" />
                        <span className="font-mono">{key}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Actions */}
                <div className="flex gap-2 mt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 h-7 text-xs"
                    onClick={(e) => {
                      e.stopPropagation()
                      onJumpToCheckpoint(checkpointId)
                    }}
                  >
                    <History className="w-3 h-3 mr-1" />
                    Jump Here
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 h-7 text-xs"
                    onClick={(e) => {
                      e.stopPropagation()
                      onForkFromCheckpoint(checkpointId)
                    }}
                  >
                    <GitBranch className="w-3 h-3 mr-1" />
                    Fork
                  </Button>
                </div>
              </div>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}
