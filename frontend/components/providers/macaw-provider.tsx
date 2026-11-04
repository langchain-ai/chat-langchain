"use client";

import type { ReactNode } from "react";
import { AppThemeProvider } from "@langchain/macaw-components/hooks/AppThemeProvider";
import { TooltipProvider } from "@langchain/macaw-components/Tooltip";

export function MacawProvider({ children }: { children: ReactNode }) {
  // Keep the existing dark-only UI and body portals in the same theme. Ignore
  // stored/system preferences until the product supports both color schemes.
  return (
    <AppThemeProvider defaultMode="dark" storageKey={null}>
      <TooltipProvider>{children}</TooltipProvider>
    </AppThemeProvider>
  );
}
