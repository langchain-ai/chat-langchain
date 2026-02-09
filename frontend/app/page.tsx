"use client";

import React from "react";
import { GraphProvider } from "./contexts/GraphContext";
import { ChatLangChain } from "./components/ChatLangChain";

export default function Page(): React.ReactElement {
  return (
    <main className="w-full h-full">
      <React.Suspense fallback={null}>
        <GraphProvider>
          <ChatLangChain />
        </GraphProvider>
      </React.Suspense>
    </main>
  );
}
