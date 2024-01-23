"use client";

import { v4 as uuidv4 } from "uuid";
import { ChatWindow } from "../app/components/ChatWindow";
import { ToastContainer } from "react-toastify";

import { ChakraProvider } from "@chakra-ui/react";

export default function Home() {
  return (
    <ChakraProvider>
      <ToastContainer />
      <ChatWindow conversationId={uuidv4()}></ChatWindow>
    </ChakraProvider>
  );
}
