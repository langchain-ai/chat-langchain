// src/app/components/RichMasterAI.tsx
import React, { useEffect, useState } from "react";
import { Box, Heading, Text, VStack, Button, Flex, useToast } from "@chakra-ui/react";
import { useTranslations } from 'next-intl';
import Login from './Login';

export function RichMasterAI() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const toast = useToast();
  const t = useTranslations('HomePage');

  useEffect(() => {
    const checkAuth = () => {
      const token = document.cookie.split(';').find(c => c.trim().startsWith('token='));
      if (token) {
        setIsAuthenticated(true);
      }
      setIsLoading(false);
    };

    checkAuth();
  }, []);

  const handleLogout = () => {
    document.cookie = 'token=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT';
    setIsAuthenticated(false);
    toast({
      title: "已退出登录",
      status: "info",
      duration: 2000,
    });
  };

  if (isLoading) {
    return (
      <Box p={8} display="flex" justifyContent="center">
        <Text>Loading...</Text>
      </Box>
    );
  }

  if (!isAuthenticated) {
    return <Login onLoginSuccess={() => setIsAuthenticated(true)} />;
  }

  return (
    <Box p={8} bg="#1A202C" color="white">
      <Flex justifyContent="flex-end" mb={4}>
        <Button
          colorScheme="red"
          size="sm"
          onClick={handleLogout}
        >
          退出登录
        </Button>
      </Flex>

      <VStack spacing={8} align="stretch">
        <Heading as="h1" size="2xl" textAlign="center">
          {t('About RichMaster Funds')}
        </Heading>
        <Text fontSize="xl" textAlign="center">
          {t('RichMaster Funds is a private fund which has had recorded stable return')}
        </Text>
      </VStack>
    </Box>
  );
}