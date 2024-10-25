// src/app/components/RichMasterAI.tsx
import React, { useEffect, useState } from "react";
import {   Box,
  Button,
  VStack,
  Heading,
  Text,
  Flex,
  Grid,
  Stat,
  StatLabel,
  StatNumber,
  StatArrow,
  SimpleGrid,
  Card,
  CardBody,
  Icon,
  Divider, useToast } from "@chakra-ui/react";
import { useTranslations } from 'next-intl';
import Login from './Login';
import {RegisterForm} from './RegisterForm'
import { FaChartLine, FaRobot, FaShieldAlt, FaLightbulb } from "react-icons/fa";

interface RichMasterAIProps {
  onRegister: () => void;
  isAuthenticated: boolean;
  setIsAuthenticated: (value: boolean) => void;
}

export function RichMasterAI({ onRegister, isAuthenticated, setIsAuthenticated }: RichMasterAIProps) {
  // const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isRegistering, setIsRegistering] = useState(false);
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
    return <Login onLoginSuccess={() => setIsAuthenticated(true)}
    onRegisterClick={() => {
      onRegister();  
    }}      
    />;
  }

  if (isRegistering) {
    return <RegisterForm onBackToLogin={() => setIsRegistering(false)} />;
  }

  return (
    <Box p={8} bg="#1A202C" color="white">
      <Flex justifyContent="flex-end" mb={2}>
        <Button
          colorScheme="red"
          size="sm"
          onClick={handleLogout}
        >
          {t('Log out')}
        </Button>
      </Flex>

      <VStack spacing={8} align="stretch" >
        <Heading as="h1" size="2xl" textAlign="center" mb={2} mt={-4} >
          {t('AI Trading Dashboard')}
        </Heading>

        {/* AI预测核心指标 */}
        <SimpleGrid columns={[1, 2, 4]} spacing={6}>
          <Stat
            bg="whiteAlpha.100"
            p={4}
            borderRadius="lg"
            boxShadow="xl"
          >
            <StatLabel>{t('Market Sentiment')}</StatLabel>
            <StatNumber color="green.400">
              <StatArrow type="increase" />
              72%
            </StatNumber>
            <Text fontSize="sm">{t('Bullish Trend')}</Text>
          </Stat>

          <Stat
            bg="whiteAlpha.100"
            p={4}
            borderRadius="lg"
            boxShadow="xl"
          >
            <StatLabel>{t('AI Confidence')}</StatLabel>
            <StatNumber color="blue.400">89%</StatNumber>
            <Text fontSize="sm">{t('Prediction Accuracy')}</Text>
          </Stat>

          <Stat
            bg="whiteAlpha.100"
            p={4}
            borderRadius="lg"
            boxShadow="xl"
          >
            <StatLabel>{t('Risk Level')}</StatLabel>
            <StatNumber color="yellow.400">Medium</StatNumber>
            <Text fontSize="sm">{t('Market Volatility')}</Text>
          </Stat>

          <Stat
            bg="whiteAlpha.100"
            p={4}
            borderRadius="lg"
            boxShadow="xl"
          >
            <StatLabel>{t('Portfolio Health')}</StatLabel>
            <StatNumber color="green.400">
              <StatArrow type="increase" />
              94%
            </StatNumber>
            <Text fontSize="sm">{t('Optimization Score')}</Text>
          </Stat>
        </SimpleGrid>

        {/* 核心功能区 */}
        <Grid templateColumns={["1fr", "1fr", "repeat(2, 1fr)"]} gap={6} mt={8}>
          <Card bg="whiteAlpha.100" boxShadow="xl">
            <CardBody>
              <Flex align="center" mb={4}>
                <Icon as={FaRobot} w={6} h={6} color="blue.400" mr={3} />
                <Heading size="md" color={"white"}>{t('AI Prediction Center')}</Heading>
              </Flex>
              <Text mb={4} color={"white"}>
                {t('Real-time market predictions with 89% accuracy using advanced ML models')}
              </Text>
              <Button colorScheme="blue" variant="outline" size="sm">
                {t('View Predictions')}
              </Button>
            </CardBody>
          </Card>

          <Card bg="whiteAlpha.100" boxShadow="xl">
            <CardBody>
              <Flex align="center" mb={4}>
                <Icon as={FaChartLine} w={6} h={6} color="green.400" mr={3} />
                <Heading size="md" color={"white"}>{t('Smart Portfolio Analysis')}</Heading>
              </Flex>
              <Text mb={4} color={"white"}>
                {t('Personalized portfolio optimization with real-time rebalancing suggestions')}
              </Text>
              <Button colorScheme="green" variant="outline" size="sm">
                {t('Analyze Portfolio')}
              </Button>
            </CardBody>
          </Card>

          <Card bg="whiteAlpha.100" boxShadow="xl">
            <CardBody>
              <Flex align="center" mb={4}>
                <Icon as={FaShieldAlt} w={6} h={6} color="purple.400" mr={3} />
                <Heading size="md" color={"white"}>{t('Risk Management')}</Heading>
              </Flex>
              <Text mb={4} color={"white"}>
                {t('Advanced risk analysis and automated hedging suggestions')}
              </Text>
              <Button colorScheme="purple" variant="outline" size="sm">
                {t('View Risk Analysis')}
              </Button>
            </CardBody>
          </Card>

          <Card bg="whiteAlpha.100" boxShadow="xl">
            <CardBody>
              <Flex align="center" mb={4}>
                <Icon as={FaLightbulb} w={6} h={6} color="yellow.400" mr={3} />
                <Heading size="md" color={"white"}>{t('Smart Alerts')}</Heading>
              </Flex>
              <Text mb={4} color={"white"}>
                {t('AI-powered market opportunity and risk alerts')}
              </Text>
              <Button colorScheme="yellow" variant="outline" size="sm">
                {t('Configure Alerts')}
              </Button>
            </CardBody>
          </Card>
        </Grid>

        <Divider my={4} borderColor="whiteAlpha.300" />

        <Box width="100%" pb={6}> {/* 添加 padding-bottom */}
          <Text 
            fontSize="lg" 
            textAlign="center" 
            maxW="800px" 
            mx="auto"
            color="whiteAlpha.900" // 确保文字颜色可见
            className="bottom-description" // 添加类名便于调试
          >
            {t('Our AI-driven platform combines advanced machine learning with real-time market data to provide accurate predictions and personalized investment strategies')}
          </Text>
        </Box>
      </VStack>
    </Box>
  );
}