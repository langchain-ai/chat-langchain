"use client";

import { ToastContainer } from "react-toastify";
import { ChakraProvider, ColorModeScript  } from "@chakra-ui/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { Client } from "@langchain/langgraph-sdk";

import { ChatWindow } from "./components/ChatWindow";
import { LangGraphClientContext } from "./hooks/useLangGraphClient";
import { API_BASE_URL, LANGCHAIN_API_KEY } from "./utils/constants";

import theme from './theme'

export default function Home() {
  const queryClient = new QueryClient();
  const langGraphClient = new Client({
    apiUrl: API_BASE_URL,
    defaultHeaders: { "x-api-key": LANGCHAIN_API_KEY },
  });
  return (
    <LangGraphClientContext.Provider value={langGraphClient}>
      <QueryClientProvider client={queryClient}>
      <ColorModeScript initialColorMode={theme.config.initialColorMode} />
        <ChakraProvider theme={theme}>
          <ToastContainer />
          <ChatWindow />
        </ChakraProvider>
      </QueryClientProvider>
    </LangGraphClientContext.Provider>
  );
}
