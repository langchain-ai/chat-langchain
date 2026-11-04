"use client";

import { useState } from "react";
import { Button } from "@langchain/macaw-components/Button";
import { ChatMessage } from "@langchain/macaw-components/ChatMessage";
import { ChatState } from "@langchain/macaw-components/ChatState";
import { Tooltip } from "@langchain/macaw-components/Tooltip";
import { useColorScheme } from "@langchain/macaw-components/hooks/useColorScheme";
import { Button as LegacyButton } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

/** Local integration fixture: no backend requests or product message handlers. */
export function MacawFoundationsSmoke() {
  const { resolvedMode } = useColorScheme();
  const [streaming, setStreaming] = useState(false);
  const [portalOpen, setPortalOpen] = useState(false);

  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-6 p-6">
      <h1 className="text-xl font-semibold">Macaw foundations</h1>
      <output aria-label="Resolved theme">Theme: {resolvedMode}</output>
      <section aria-label="Macaw components" className="flex flex-col gap-4 rounded-lg border p-4">
        <div className="flex flex-wrap gap-3">
          <Button onClick={() => setStreaming((value) => !value)}>
            Toggle message state
          </Button>
          <Tooltip title="Macaw portal inherits the document theme" open={portalOpen}>
            <Button color="secondary" onClick={() => setPortalOpen((value) => !value)}>
              Toggle portal
            </Button>
          </Tooltip>
        </div>
        <ChatMessage
          name="Assistant"
          message={{
            id: "foundation-message",
            role: "assistant",
            status: streaming ? "streaming" : "completed",
            content: [{ id: "foundation-body", type: "text", text: "Macaw chat content uses the published styles and fonts." }],
          }}
        />
        <ChatState status={streaming ? "streaming" : "completed"} label={streaming ? "Streaming" : "Complete"} />
      </section>
      <section aria-label="Legacy controls" className="flex flex-col gap-4 rounded-lg border border-chat-border bg-chat-card p-4 text-chat-foreground">
        <p className="text-chat-xs text-chat-muted-foreground">Legacy controls keep their existing palette and small text size.</p>
        <LegacyButton>Legacy primary action</LegacyButton>
        <label htmlFor="legacy-input">Legacy input</label>
        <Input id="legacy-input" placeholder="Check keyboard focus" />
        <label htmlFor="legacy-textarea">Legacy textarea</label>
        <Textarea id="legacy-textarea" placeholder="Check keyboard focus" />
      </section>
    </main>
  );
}
