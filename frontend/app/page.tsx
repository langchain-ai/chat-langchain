"use client";

import { ToastContainer } from "react-toastify";
import { ChakraProvider } from "@chakra-ui/react";
import { QueryClient, QueryClientProvider } from "react-query";

import { ChatWindow } from "./components/ChatWindow";

export default function Home() {
  const queryClient = new QueryClient()
  return (
    <QueryClientProvider client={queryClient}>
      <ChakraProvider>
        <ToastContainer />
        <ChatWindow />
      </ChakraProvider>
    </QueryClientProvider>
  );
}
