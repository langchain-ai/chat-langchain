"use client";

import { ChatWindow } from "./components/ChatWindow";
import { ToastContainer } from "react-toastify";

import { ChakraProvider } from "@chakra-ui/react";

export default function Home() {
  return (
    <ChakraProvider>
      <ToastContainer />
      <ChatWindow />
    </ChakraProvider>
  );
}
