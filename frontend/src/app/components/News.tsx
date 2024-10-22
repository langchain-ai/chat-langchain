// In News.tsx
import React, { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { Box, Heading, Text, VStack, Image, Flex,Link } from "@chakra-ui/react";
import { useLocale,useTranslations } from "next-intl";

interface NewsData {
    flashNewsGuid: string;
    flashNewsLink: string;
    generateDate: string;
    id: number;
    language: string;
    pubDate: string;
    slug: string;
    sourceId: number;
    status: string;
    summary: string;
    title: string;
  }

  export function News() {
    const [news, setNews] = useState<NewsData[]>([]);
    const [page, setPage] = useState<number>(0); // 用于控制加载的页数
    const [loading, setLoading] = useState<boolean>(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const locale = useLocale();
    const t = useTranslations('HomePage');
  
    const fetchNews = useCallback(async () => {
      setLoading(true);
      try {
        const response = await fetch(
          "https://richmaster-aubpechkfafyegfy.eastasia-01.azurewebsites.net/api/flashnews?language=${locale}"
        );
  
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
  
        const data: NewsData[] = await response.json();
      // 手动过滤新闻数据，依据语言
      const filteredData = data.filter(item => item.language === locale);
      setNews((prevNews) => [...prevNews, ...filteredData.slice(page * 30, (page + 1) * 30)]);
    } catch (error) {
      console.error("Error fetching the news:", error);
    } finally {
      setLoading(false);
    }
  }, [page, locale]);
  
    useEffect(() => {
      fetchNews();
    }, [fetchNews]);
  
    useEffect(() => {
      const handleScroll = () => {
        if (!loading && containerRef.current) {
          const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
          if (scrollTop + clientHeight >= scrollHeight - 5) {
            setPage((prevPage) => prevPage + 1);
          }
        }
      };
  
      const container = containerRef.current;
      if (container) {
        container.addEventListener("scroll", handleScroll);
      }
      return () => {
        if (container) {
          container.removeEventListener("scroll", handleScroll);
        }
      };
    }, [loading]);
  
    return (
      <Box ref={containerRef} p={8} bg="#1A202C" color="white" minHeight="100vh" overflowY="auto">
        <VStack spacing={8} align="stretch">
          <Heading as="h1" size="2xl" textAlign="center" mb={8}>
          {t('Latest News')}
            
          </Heading>
  
          {news.map((item) => (
            <Box
              key={item.flashNewsGuid}
              p={5}
              shadow="md"
              borderWidth="1px"
              bg="#2D3748"
              borderRadius="md"
            >
              <Flex direction="column" gap={3}>
                <Text fontSize="lg" fontWeight="bold" mb={2}>
                  {item.title}
                </Text>
                <Text fontSize="md">{item.summary}</Text>
                <Link href={item.flashNewsLink} isExternal color="teal.300">
                  Read more
                </Link>
              </Flex>
            </Box>
          ))}
  
          {loading && <Text textAlign="center">Loading more news...</Text>}
        </VStack>
      </Box>
    );
  }