"use client";

import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { List, ListItem, Spacer, Text, Button, visuallyHiddenStyle, VStack } from "@chakra-ui/react";
import { PlusSquareIcon, DeleteIcon, ChatIcon, InfoIcon, SunIcon, LockIcon, CheckIcon } from "@chakra-ui/icons";

import { ThreadListProps } from "../hooks/useThreadList";
import { useThread } from "../hooks/useThread";
import { Vast_Shadow } from "next/font/google";

export function ChatList(props: {
  userId: string;
  threads: ThreadListProps["threads"];
  areThreadsLoading: ThreadListProps["areThreadsLoading"];
  loadMoreThreads: ThreadListProps["loadMoreThreads"];
  enterChat: (id: string | null) => void;
  deleteChat: (id: string) => void;
  enterRegister: () => void;
  enterAboutUs: () => void;
  enterRichMasterFunds: () => void;
  enterPricingPlan: () => void;
  enterPreviousChats: () => void;
}) {
  const { currentThread } = useThread(props.userId);
  const [isRegistering, setIsRegistering] = useState(false);
  const [isAboutUs, setIsAboutUs] = useState(false);
  const [isRichMasterFunds, setIsRichMasterFunds] = useState(false);
  const [isPricingPlan, setIsPricingPlan] = useState(false);
  const [isPreviousChats, setIsPreviousChats] = useState(false);


  const [prevScrollTop, setPrevScrollTop] = useState(0);
  const listRef = useRef<HTMLUListElement>(null);
  const handleScroll = useCallback(() => {
    if (!listRef.current || props.areThreadsLoading) {
      return;
    }

    const { scrollTop, scrollHeight, clientHeight } = listRef.current;
    const isScrollingDown = scrollTop > prevScrollTop;
    setPrevScrollTop(scrollTop);

    if (isScrollingDown && scrollTop + clientHeight >= scrollHeight - 5) {
      props.loadMoreThreads();
    }
  }, [prevScrollTop, props.areThreadsLoading]);

  useEffect(() => {
    const listElement = listRef.current;
    if (listElement) {
      listElement.addEventListener("scroll", handleScroll);
    }
    return () => {
      if (listElement) {
        listElement.removeEventListener("scroll", handleScroll);
      }
    };
  }, [props.areThreadsLoading]);

  return (
    <VStack spacing={4} align="stretch">
      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(68,70,84)" }}
        onClick={() => {
          props.enterChat(null);
          setIsRegistering(false);  // <-- Reset registration state when starting a new chat
          setIsAboutUs(false)
          setIsRichMasterFunds(false);
          setIsPricingPlan(false);
          setIsPreviousChats(false)
        }}
      >
        <PlusSquareIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>New chat</Text>
      </Button>
  
      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={() => {
          props.enterPreviousChats();
          setIsRegistering(false);
          setIsAboutUs(false);
          setIsRichMasterFunds(false);
          setIsPricingPlan(false);
          setIsPreviousChats(false)
        }}
      >
        <ChatIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>Previous chats</Text>
      </Button>

      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={props.enterRichMasterFunds}
        mt={4}
      >
        <SunIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>RichMaster Fund</Text>
      </Button>
      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={props.enterPricingPlan}
        mt={4}
      >
         <LockIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>Pricing Plan</Text>
      </Button>

      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={props.enterRegister}
        mt={4}
      >
        <CheckIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>Register</Text>
      </Button>

      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={props.enterAboutUs}
        mt={4}
      >
        <InfoIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>About Us</Text>
      </Button>
      
    </VStack>
  );
}
