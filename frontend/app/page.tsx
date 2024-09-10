"use client";

import { ToastContainer } from "react-toastify";
import { ChakraProvider, ColorModeScript  } from "@chakra-ui/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { Client } from "@langchain/langgraph-sdk";

import { ChatWindow } from "./components/ChatWindow";
import { LangGraphClientContext } from "./hooks/useLangGraphClient";

const apiUrl = process.env.NEXT_PUBLIC_API_URL
  ? process.env.NEXT_PUBLIC_API_URL
  : "http://localhost:3000/api";

import theme from './theme'

export default function Home() {
  const queryClient = new QueryClient();
  const langGraphClient = new Client({
    apiUrl,
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
