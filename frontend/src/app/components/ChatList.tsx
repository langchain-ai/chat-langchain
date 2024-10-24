"use client";

import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { List, ListItem, Spacer, Text, Button, visuallyHiddenStyle, VStack,Box, Select,Flex } from "@chakra-ui/react";
import { GiArtificialIntelligence } from "react-icons/gi";
import { PlusSquareIcon, DeleteIcon, ChatIcon, InfoIcon, SunIcon, LockIcon, CheckIcon } from "@chakra-ui/icons";

import { ThreadListProps } from "../hooks/useThreadList";
import { useThread } from "../hooks/useThread";
import { Vast_Shadow } from "next/font/google";
import { useTranslations, useLocale } from 'next-intl';
import { useRouter, usePathname, Link } from '@/src/i18n/routing';

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
  enterNews: () => void;
  enterRichMasterAI:()=> void;
}) {

    const router = useRouter();
  const pathname = usePathname();
  const locale = useLocale();  // 获取当前语言

  const handleLocaleChange = (newLocale: string) => {
    // router.replace(`/${newLocale}${pathname}`); // 确保切换时 URL 正确
    window.location.href =  `/${newLocale}${pathname}`;
  };

  const { currentThread } = useThread(props.userId);
  const [isRegistering, setIsRegistering] = useState(false);
  const [isAboutUs, setIsAboutUs] = useState(false);
  const [isRichMasterFunds, setIsRichMasterFunds] = useState(false);
  const [isPricingPlan, setIsPricingPlan] = useState(false);
  const [isPreviousChats, setIsPreviousChats] = useState(false);
  const [isNews, setIsNews] = useState(false);
  const [isRichMasterAI, setIsRichMasterAI] = useState(false);

  

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

  const t = useTranslations('HomePage'); 
  return (
    <Flex direction="column" height="100vh" alignItems="stretch">
    <VStack spacing={4} align="stretch" >
      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(68,70,84)" }}
        onClick={() => {
          props.enterChat(null);
          setIsRegistering(false);  // <-- Reset registration state when starting a new chat
          setIsAboutUs(false)
          setIsRichMasterFunds(false);
          setIsPricingPlan(false);
          setIsNews(false);
          setIsPreviousChats(false);
          setIsRichMasterAI(false)
        }}
      >
        <PlusSquareIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>{t('New chat')}</Text>
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
          setIsNews(false);
          setIsPreviousChats(false)
        }}
      >
        <ChatIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>{t('Previous chats')}</Text>
      </Button>

      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={props.enterRichMasterFunds}
        mt={4}
      >
        <SunIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>{t('RichMaster Fund')}</Text>
      </Button>

      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={props.enterRichMasterAI}
        mt={4}
      >
        <GiArtificialIntelligence color={"white"} style={{marginRight:'4px'}} />
        <Text color={"white"}>{t('RichMasterAI')}</Text>
      </Button>

      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={props.enterPricingPlan}
        mt={4}
      >
         <LockIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>{t('Pricing Plan')}</Text>
      </Button>

      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={props.enterRegister}
        mt={4}
      >
        <CheckIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>{t('Register')}</Text>
      </Button>

      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={props.enterAboutUs}
        mt={4}
      >
        <InfoIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>{t('about')}</Text>
      </Button>

      <Button
        backgroundColor={"#2a4365"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={props.enterNews}
        mt={4}
      >
        <InfoIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>{t('News')}</Text>
      </Button>
  </VStack>
      
       {/* Language Selector */}
          <Box p={4} mt="auto" mb={16} width="100%">
        <Select
          value={locale}
          onChange={(e) => handleLocaleChange(e.target.value)}
          backgroundColor="#2a4365"
          color="white"
          _hover={{ backgroundColor: "rgb(78,78,81)" }}
          iconColor="white"
          textAlign="center"
        >
          <option style={{ backgroundColor: "#2a4365", color: "white",textAlign:"center" }} value="en">English</option>
          <option style={{ backgroundColor: "#2a4365", color: "white",textAlign:"center" }} value="zh">中文</option>
          {/* Add more languages as needed */}
        </Select>
      </Box>  
  
      </Flex>
  );
}
