"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "react-toastify";
import { useRouter, useSearchParams } from "next/navigation";
import "highlight.js/styles/gradient-dark.css";
import "react-toastify/dist/ReactToastify.css";
import {
  Heading,
  Flex,
  IconButton,
  InputGroup,
  InputRightElement,
  Spinner,
  Button,
  Text,
  Box,
  useBreakpointValue,
  VStack,
} from "@chakra-ui/react";
import { ArrowDownIcon, ArrowUpIcon, SmallCloseIcon, HamburgerIcon, InfoOutlineIcon, CloseIcon } from "@chakra-ui/icons";
import { Select, Link } from "@chakra-ui/react";
import { Client } from "@langchain/langgraph-sdk";
import { v4 as uuidv4 } from "uuid";

import { EmptyState } from "./EmptyState";
import { ChatMessageBubble } from "./ChatMessageBubble";
import { ChatList } from "./ChatList";
import { AutoResizeTextarea } from "./AutoResizeTextarea";
import { Message } from "../types";
import { useThread } from "../hooks/useThread";
import { useThreadList } from "../hooks/useThreadList";
import { useThreadMessages } from "../hooks/useThreadMessages";
import { useLangGraphClient } from "../hooks/useLangGraphClient";
import { useStreamState } from "../hooks/useStreamState";
import { useLocalStorage } from "../hooks/useLocalStorage";

const MODEL_TYPES = ["openai_gpt_4o_mini", "anthropic_claude_3_haiku"];

const defaultLlmValue =
  MODEL_TYPES[Math.floor(Math.random() * MODEL_TYPES.length)];

const getAssistantId = async (client: Client) => {
  const response = await client.assistants.search({
    metadata: null,
    offset: 0,
    limit: 10,
    graphId: "chat",
  });
  if (response.length !== 1) {
    throw Error(
      `Expected exactly one assistant, got ${response.length} instead`,
    );
  }
  return response[0]["assistant_id"];
};

// // Add mock stock data
// const mockStockData = [
//   { name: "Dow Jones", value: "34,721.91", change: "+0.30%" },
//   { name: "NASDAQ", value: "15,785.32", change: "-0.15%" },
//   { name: "S&P 500", value: "4,509.23", change: "+0.20%" },
//   { symbol: "AAPL", price: "150.25", change: "+1.25%" },
//   { symbol: "MSFT", price: "305.15", change: "+0.75%" },
//   { symbol: "GOOGL", price: "2750.80", change: "-0.20%" },
//   { symbol: "AMZN", price: "3380.45", change: "+1.50%" },
// ];
interface StockData {
  name?: string;
  value?: string;
  symbol: string;
  price: string | number;
  change: string | number;
}

type SetStockDataFunction = React.Dispatch<React.SetStateAction<StockData[]>>;
// // Add StockPanel component
// interface StockPanelProps {
//   isVisible: boolean;
//   onClose: () => void;
// }

// const StockPanel: React.FC<StockPanelProps> = ({ isVisible, onClose }) => (
//   <Box
//     position="fixed"
//     right="0"
//     top="0"
//     bottom="0"
//     width="300px"
//     bg="#2D3748"
//     p={4}
//     overflowY="auto"
//     transform={isVisible ? "translateX(0)" : "translateX(100%)"}
//     transition="transform 0.3s"
//     zIndex="5"
//   >
//     <Flex justify="space-between" align="center" mb={4}>
//       <Heading size="md" color="white">Stock Info</Heading>
//       <IconButton
//         icon={<CloseIcon />}
//         onClick={onClose}
//         aria-label="Close stock panel"
//         size="sm"
//         variant="ghost"
//         color="white"
//       />
//     </Flex>
//     <VStack align="stretch" spacing={4}>
//       <Box>
//         <Heading size="sm" color="white" mb={2}>Market Indices</Heading>
//         {mockStockData.slice(0, 3).map((index) => (
//           <Flex key={index.name} justify="space-between" color="white">
//             <Text>{index.name}</Text>
//             <Text>{index.value}</Text>
//             <Text color={index.change.startsWith('+') ? "green.300" : "red.300"}>{index.change}</Text>
//           </Flex>
//         ))}
//       </Box>
//       <Box>
//         <Heading size="sm" color="white" mb={2}>Popular Stocks</Heading>
//         {mockStockData.slice(3).map((stock) => (
//           <Flex key={stock.symbol} justify="space-between" color="white">
//             <Text>{stock.symbol}</Text>
//             <Text>{stock.price}</Text>
//             <Text color={stock.change.startsWith('+') ? "green.300" : "red.300"}>{stock.change}</Text>
//           </Flex>
//         ))}
//       </Box>
//     </VStack>
//   </Box>
// );

// Replace with your actual stock API URL and key
const STOCK_API_URL = 'https://yfinance-fza5dthrg6dxd2c3.southeastasia-01.azurewebsites.net/';
const STOCK_API_KEY = 'STOCK_API_KEY';

const VALID_STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"];

// Initial mock data
const initialMockStockData: StockData[] = [
  { name: "Dow Jones", value: "34,721.91", change: "+0.50%", symbol: "DJI", price: "N/A" },
  { name: "NASDAQ", value: "15,785.32", change: "-0.15%", symbol: "IXIC", price: "N/A" },
  { name: "S&P 500", value: "4,509.23", change: "+0.20%", symbol: "SPX", price: "N/A" },
  { symbol: "AAPL", price: "150.25", change: "+1.25%" },
  { symbol: "MSFT", price: "305.15", change: "+0.75%" },
  { symbol: "GOOGL", price: "2750.80", change: "-0.20%" },
  { symbol: "AMZN", price: "3380.45", change: "+1.50%" },
];

const fetchStockDataFromApi = async (symbol: string): Promise<StockData | null> => {
  try {
    const response = await fetch(`${STOCK_API_URL}/stock?symbol=${symbol}`, {
      method: 'GET',
      headers: {
        'X-API-KEY': STOCK_API_KEY,
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) throw new Error('Network response was not ok');

    const data = await response.json();
    return {
      symbol: data.symbol,
      price: data.latestPrice,
      change: data.changePercent,
    };
  } catch (error) {
    console.error('Fetch stock data failed:', error);
    return null;
  }
};

// Function to update stock data
const fetchAndUpdateStockData = async (setStockData: SetStockDataFunction) => {
  const updates = await Promise.all(
    VALID_STOCK_SYMBOLS.map(async symbol => {
      const data = await fetchStockDataFromApi(symbol);
      return data || { symbol, price: 'N/A', change: 'N/A' };
    })
  );
  
  setStockData(prevData => [
    ...prevData.slice(0, 3),  // Keep the first 3 items (market indices)
    ...updates
  ]);
};

interface StockPanelProps {
  isVisible: boolean;
  onClose: () => void;
}

const StockPanel: React.FC<StockPanelProps> = ({ onClose }) => {
  const [stockData, setStockData] = useState<StockData[]>(initialMockStockData);

  useEffect(() => {
    fetchAndUpdateStockData(setStockData);

    const intervalId = setInterval(() => fetchAndUpdateStockData(setStockData), 3000);

    return () => clearInterval(intervalId);
  }, []);

  return (
    <Box
      position="fixed"
      right="0"
      top="0"
      bottom="0"
      width="300px"
      bg="#2D3748"
      p={4}
      overflowY="auto"
      transform="translateX(0)"
      transition="transform 0.3s"
      zIndex="5"
    >
      <VStack align="stretch" spacing={4}>
        <Box>
          <Heading size="sm" color="white" mb={2}>Market Indices</Heading>
          {stockData.slice(0, 3).map((index) => (
            <Flex key={index.name} justify="space-between" color="white">
              <Text>{index.name}</Text>
              <Text>{index.value}</Text>
   <Text color={parseFloat((index.change as string).replace('%', '')) > 0 ? "green.300" : "red.300"}>{index.change}%</Text>
       
            </Flex>
          ))}
        </Box>
        <Box>
          <Heading size="sm" color="white" mb={2}>Popular Stocks</Heading>
          {stockData.slice(3).map((stock) => (
            <Flex key={stock.symbol} justify="space-between" color="white">
              <Text>{stock.symbol}</Text>
              <Text>{stock.price}</Text>
              <Text color={parseFloat((stock.change as string).replace('%', '')) > 0 ? "green.300" : "red.300"}>{stock.change}%</Text> 
             

            </Flex>
          ))}
        </Box>
      </VStack>
    </Box>
  );
};


export function ChatWindow() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  const newestMessageRef = useRef<HTMLDivElement>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [llm, setLlm] = useState(
    searchParams.get("llm") ?? "openai_gpt_4o_mini",
  );
  const [llmIsLoading, setLlmIsLoading] = useState(true);
  const [assistantId, setAssistantId] = useState<string>("");
  const [userId, setUserId] = useLocalStorage("userId", null);
  const [isChatListVisible, setIsChatListVisible] = useState(false);
  const [isStockPanelVisible, setIsStockPanelVisible] = useState(false);

  const isMobile = useBreakpointValue({ base: true, md: false });

  const client = useLangGraphClient();

  const { currentThread } = useThread(userId);
  const {
    threads,
    createThread,
    updateThread,
    deleteThread,
    loadMoreThreads,
    areThreadsLoading,
  } = useThreadList(userId);
  const { streamStates, startStream, stopStream } = useStreamState();
  const streamState =
    currentThread == null
      ? null
      : streamStates[currentThread.thread_id] ?? null;
  const { refreshMessages, messages, setMessages, next, areMessagesLoading } =
    useThreadMessages(
      currentThread?.thread_id ?? null,
      streamState,
      stopStream,
    );

  const scrollToNewestMessage = useCallback(() => {
    if (newestMessageRef.current) {
      newestMessageRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  useEffect(() => {
    if (messages.length > 0) {
      scrollToNewestMessage();
    }
  }, [messages, scrollToNewestMessage]);

  const setLanggraphInfo = async () => {
    try {
      const assistantId = await getAssistantId(client);
      setAssistantId(assistantId);
    } catch (e) {
      toast.error("Could not load AI agent");
    }
  };

  const setUserInfo = () => {
    if (userId == null) {
      const userId = uuidv4();
      setUserId(userId);
    }
  };

  useEffect(() => {
    setLlm(searchParams.get("llm") ?? defaultLlmValue);
    setUserInfo();
    setLanggraphInfo();
    setLlmIsLoading(false);
  }, []);

  const config = {
    configurable: { model_name: llm },
    tags: ["model:" + llm],
  };

  const getThreadName = (messageValue: string) =>
    messageValue.length > 20 ? messageValue.slice(0, 20) + "..." : messageValue;

  const renameThread = async (messageValue: string) => {
    if (currentThread == null || messages.length > 1) {
      return;
    }
    const threadName = getThreadName(messageValue);
    await updateThread(currentThread["thread_id"], threadName);
  };

  const sendMessage = async (message?: string) => {
    if (messageContainerRef.current) {
      messageContainerRef.current.classList.add("grow");
    }
    if (isLoading) {
      return;
    }

    const messageValue = message ?? input;
    if (messageValue === "") return;

    let thread = currentThread;
    if (thread == null) {
      const threadName = getThreadName(messageValue);
      thread = await createThread(threadName);
      insertUrlParam("threadId", thread["thread_id"]);
    }

    setInput("");
    const formattedMessage: Message = {
      id: Math.random().toString(),
      content: messageValue,
      type: "human",
    };
    setMessages((prevMessages) => [...prevMessages, formattedMessage]);
    setIsLoading(true);

    try {
      await renameThread(messageValue);
      await startStream(
        [formattedMessage],
        thread["thread_id"],
        assistantId,
        config,
      );
      await refreshMessages();
      setIsLoading(false);
    } catch (e) {
      setIsLoading(false);
      if (!(e instanceof DOMException && e.name == "AbortError")) {
        throw e;
      }
    }
  };

  const sendInitialQuestion = async (question: string) => {
    await sendMessage(question);
  };

  const continueStream = async (threadId: string) => {
    try {
      setIsLoading(true);
      await startStream(null, threadId, assistantId, config);
      setIsLoading(false);
    } catch (e) {
      setIsLoading(false);
      if (!(e instanceof DOMException && e.name == "AbortError")) {
        throw e;
      }
    }
  };

  const insertUrlParam = (key: string, value?: string) => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set(key, value ?? "");
    const newUrl =
      window.location.protocol +
      "//" +
      window.location.host +
      window.location.pathname +
      "?" +
      searchParams.toString();
    router.push(newUrl);
  };

  const selectThread = useCallback(
    async (id: string | null) => {
      if (!id) {
        const thread = await createThread("New chat");
        insertUrlParam("threadId", thread["thread_id"]);
      } else {
        insertUrlParam("threadId", id);
      }
      setIsChatListVisible(false);
    },
    [setMessages, createThread, insertUrlParam],
  );

  const deleteThreadAndReset = async (id: string) => {
    await deleteThread(id);
    router.push(
      window.location.protocol +
        "//" +
        window.location.host +
        window.location.pathname,
    );
  };

  const toggleChatList = () => setIsChatListVisible(!isChatListVisible);
  const toggleStockPanel = () => setIsStockPanelVisible(!isStockPanelVisible);

  return (
    <Flex direction="column" h="100vh" overflow="hidden" w="full">
      {isMobile && (
        <IconButton
          aria-label="Toggle chat list"
          icon={<HamburgerIcon />}
          onClick={toggleChatList}
          position="absolute"
          top="4"
          left="4"
          zIndex="10"
        />
      )}
      
      <Flex direction={["column", "column", "row"]} h="full">
        <Box
          w={["full", "full", "212px"]}
          minW={["auto", "auto", "212px"]}
          pt={["4", "4", "36px"]}
          px={["4", "4", "24px"]}
          overflowY="auto"
          position={["fixed", "fixed", "relative"]}
          left="0"
          top="0"
          bottom="0"
          bg="gray.800"
          zIndex="5"
          transform={[
            isChatListVisible ? "translateX(0)" : "translateX(-100%)",
            isChatListVisible ? "translateX(0)" : "translateX(-100%)",
            "translateX(0)"
          ]}
          transition="transform 0.3s"
        >
          <ChatList
            userId={userId}
            threads={threads}
            enterChat={selectThread}
            deleteChat={deleteThreadAndReset}
            areThreadsLoading={areThreadsLoading}
            loadMoreThreads={loadMoreThreads}
          />
        </Box>

        <Flex direction="column" flex="1" overflow="hidden" w="full" pt={isMobile ? "12" : "4"}>
          <Flex direction="column" alignItems="center" mb="4">
            <Heading
              fontSize={["xl", "2xl", "3xl"]}
              fontWeight="medium"
              mb="1"
              color="white"
              textAlign="center"
            >
              RichMaster StockGPT 🪙
            </Heading>
          </Flex>
          <IconButton
            aria-label="Toggle stock panel"
            icon={<InfoOutlineIcon />}
            onClick={toggleStockPanel}
            position="absolute"
            top="4"
            right="4"
            zIndex="10"
          />
          {areMessagesLoading ? (
            <Spinner alignSelf="center" my="2" />
          ) : (
            <Flex direction="column" flex="1" overflow="hidden">
              <Box
                ref={messageContainerRef}
                flex="1"
                overflowY="auto"
                mb="2"
                maxH={["60vh", "60vh", "calc(100vh - 300px)"]}
                px={4}
              >
                {messages.length > 0 && currentThread != null ? (
                  <Flex direction="column-reverse">
                    {next.length > 0 && streamStates[currentThread.thread_id]?.status !== "inflight" && (
                      <Button
                        key="continue-button"
                        backgroundColor="rgb(58, 58, 61)"
                        _hover={{ backgroundColor: "rgb(78,78,81)" }}
                        onClick={() => continueStream(currentThread["thread_id"])}
                        mb="2"
                      >
                        <ArrowDownIcon color="white" mr="2" />
                        <Text color="white">Click to continue</Text>
                      </Button>
                    )}
                    {[...messages].reverse().map((m, index) => (
                      <Box 
                        key={m.id} 
                        ref={index === 0 ? newestMessageRef : null}
                      >
                        <ChatMessageBubble
                          message={{ ...m }}
                          feedbackUrls={streamStates[currentThread.thread_id]?.feedbackUrls}
                          aiEmoji="🦜"
                          isMostRecent={index === 0}
                          messageCompleted={!isLoading}
                        />
                      </Box>
                    ))}
                  </Flex>
                ) : (
                  <EmptyState onChoice={sendInitialQuestion} />
                )}
              </Box>
              <Flex 
                direction="column" 
                alignItems="center" 
                width="100%" 
                maxW={["100%", "100%", "800px"]} 
                mx="auto" 
                mt="auto" 
                pb={4}
              >
                <InputGroup size="md" alignItems="center" width="100%">
                  <AutoResizeTextarea
                    value={input}
                    maxRows={5}
                    mr="56px"
                    placeholder="Discover stocks now"
                    textColor="white"
                    borderColor="rgb(58, 58, 61)"
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      } else if (e.key === "Enter" && e.shiftKey) {
                        e.preventDefault();
                        setInput(input + "\n");
                      }
                    }}
                  />
                  <InputRightElement h="full">
                    <IconButton
                      colorScheme="blue"
                      rounded="full"
                      aria-label="Send"
                      icon={isLoading ? <SmallCloseIcon /> : <ArrowUpIcon />}
                      type="submit"
                      onClick={(e) => {
                        e.preventDefault();
                        if (currentThread != null && isLoading) {
                          stopStream?.(currentThread.thread_id);
                        } else {
                          sendMessage();
                        }
                      }}
                    />
                  </InputRightElement>
                </InputGroup>
                {messages.length === 0 && (
                  <Text as="footer" textAlign="center" mt={2} color="white">
                    Start trading today!
                  </Text>
                )}
              </Flex>
            </Flex>
          )}
        </Flex>
      </Flex>
{/*       { isStockPanelVisible &&  
      <StockPanel  onClose={() => setIsStockPanelVisible(false)} />} */}
      <StockPanel isVisible={isStockPanelVisible} onClose={() => setIsStockPanelVisible(false)} />
    </Flex>
  );
}
