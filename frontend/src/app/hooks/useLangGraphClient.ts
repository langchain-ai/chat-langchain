"use client";

import { Client } from "@langchain/langgraph-sdk";
import { createContext, useContext } from "react";

export const LangGraphClientContext = createContext<Client | undefined>(
  undefined,
);

export const useLangGraphClient = (langGraphClient?: Client) => {
  const client = useContext(LangGraphClientContext);

  if (langGraphClient) {
    return langGraphClient;
  }

  if (!client) {
    throw new Error(
      "No LangGraphClient set, use LangGraphClientContext to set one",
    );
  }

  return client;
};
