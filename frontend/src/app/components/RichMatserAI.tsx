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
  Divider, useToast,  Tabs, TabList, TabPanels, Tab, TabPanel,
  Select, FormControl, FormLabel, NumberInput,
  NumberInputField, NumberInputStepper,
  NumberIncrementStepper, NumberDecrementStepper,
  Stack, 
  Table,
  Tr,
  Td,
  StatHelpText,
  WrapItem,
  Wrap,
  Tbody,
  InputGroup,
  InputRightElement,
  Input,
  HStack,
  Progress,
  ListItem,
  List,
  Tag,
  Switch,
  Slider,
  CheckboxGroup,
  SliderTrack,
  Checkbox,
  SliderThumb,
  SliderFilledTrack,
  Spacer,
  IconButton,
  Thead,
  Th} from "@chakra-ui/react";
import { useTranslations } from 'next-intl';
import Login from './Login';
import {RegisterForm} from './RegisterForm'
import { FaChartLine, FaRobot, FaShieldAlt, FaLightbulb, FaArrowUp, FaMinus, FaArrowDown, FaChartArea, FaChartBar, FaTimes } from "react-icons/fa";

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
  const [tabIndex, setTabIndex] = useState(0);

  const handleViewPredictions = () => {
    setTabIndex(2); // Prediction tab index
  };

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

  const renderDashboard = () => (
    <VStack 
      spacing={8} 
      align="stretch"
      px={{ base: 2, md: 0 }}  // 移动端添加横向内边距
      overflowY="auto"
      maxHeight={{ base: "calc(100vh - 150px)", md: "auto" }}  // 移动端可滚动
    >
      {/* AI预测核心指标 */}
      <SimpleGrid columns={[1, 2, 4]} spacing={{ base: 4, md: 6 }}>
        <Stat
          bg="whiteAlpha.100"
          p={{ base: 3, md: 4 }}  // 移动端减小内边距
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
          p={{ base: 3, md: 4 }}
          borderRadius="lg"
          boxShadow="xl"
        >
          <StatLabel>{t('AI Confidence')}</StatLabel>
          <StatNumber color="blue.400">89%</StatNumber>
          <Text fontSize="sm">{t('Prediction Accuracy')}</Text>
        </Stat>
  
        <Stat
          bg="whiteAlpha.100"
          p={{ base: 3, md: 4 }}
          borderRadius="lg"
          boxShadow="xl"
        >
          <StatLabel>{t('Risk Level')}</StatLabel>
          <StatNumber color="yellow.400">Medium</StatNumber>
          <Text fontSize="sm">{t('Market Volatility')}</Text>
        </Stat>
  
        <Stat
          bg="whiteAlpha.100"
          p={{ base: 3, md: 4 }}
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
      <Grid 
        templateColumns={["1fr", "1fr", "repeat(2, 1fr)"]} 
        gap={{ base: 4, md: 6 }} 
        mt={8}
      >
        <Card bg="whiteAlpha.100" boxShadow="xl">
          <CardBody p={{ base: 4, md: 6 }}>  {/* 移动端减小内边距 */}
            <Flex align="center" mb={4}>
              <Icon as={FaRobot} w={6} h={6} color="blue.400" mr={3} />
              <Heading size="md" color={"white"}>{t('AI Prediction Center')}</Heading>
            </Flex>
            <Text mb={4} color={"white"}>
              {t('Real-time market predictions with 89% accuracy using advanced ML models')}
            </Text>
            <Button colorScheme="blue" variant="outline" size="sm" onClick={handleViewPredictions}>
              {t('View Predictions')}
            </Button>
          </CardBody>
        </Card>
  
        <Card bg="whiteAlpha.100" boxShadow="xl">
          <CardBody p={{ base: 4, md: 6 }}>
            <Flex align="center" mb={4}>
              <Icon as={FaChartLine} w={6} h={6} color="green.400" mr={3} />
              <Heading size="md" color={"white"}>{t('Smart Portfolio Analysis')}</Heading>
            </Flex>
            <Text mb={4} color={"white"}>
              {t('Personalized portfolio optimization with real-time rebalancing suggestions')}
            </Text>
            <Button colorScheme="green" variant="outline" size="sm" onClick={() => setTabIndex(3)} >
              {t('Analyze Portfolio')}
            </Button>
          </CardBody>
        </Card>
  
        <Card bg="whiteAlpha.100" boxShadow="xl">
          <CardBody p={{ base: 4, md: 6 }}>
            <Flex align="center" mb={4}>
              <Icon as={FaShieldAlt} w={6} h={6} color="purple.400" mr={3} />
              <Heading size="md" color={"white"}>{t('Risk Management')}</Heading>
            </Flex>
            <Text mb={4} color={"white"}>
              {t('Advanced risk analysis and automated hedging suggestions')}
            </Text>
            <Button colorScheme="purple" variant="outline" size="sm" onClick={() => setTabIndex(4)}>
              {t('View Risk Analysis')}
            </Button>
          </CardBody>
        </Card>
  
        <Card bg="whiteAlpha.100" boxShadow="xl">
          <CardBody p={{ base: 4, md: 6 }}>
            <Flex align="center" mb={4}>
              <Icon as={FaLightbulb} w={6} h={6} color="yellow.400" mr={3} />
              <Heading size="md" color={"white"}>{t('Smart Alerts')}</Heading>
            </Flex>
            <Text mb={4} color={"white"}>
              {t('AI-powered market opportunity and risk alerts')}
            </Text>
            <Button colorScheme="yellow" variant="outline" size="sm" onClick={()=> setTabIndex(5)}>
              {t('Configure Alerts')}
            </Button>
          </CardBody>
        </Card>
      </Grid>
  
      <Divider my={4} borderColor="whiteAlpha.300" />
  
      <Box width="100%" pb={6}>
        <Text 
          fontSize={{ base: "md", md: "lg" }}  // 移动端减小字体
          textAlign="center" 
          maxW="800px" 
          mx="auto"
          px={{ base: 4, md: 0 }}  // 移动端添加横向内边距
          color="whiteAlpha.900"
          className="bottom-description"
        >
          {t('RichMasterAIIntro')}
        </Text>
      </Box>
    </VStack>
  );

  const renderSettings = () => (
    <Box 
      height={{ base: "calc(100vh - 120px)", md: "calc(100vh - 150px)" }} 
      overflowY="auto" 
      px={{ base: 2, md: 2 }}
    >
      <VStack spacing={{ base: 4, md: 6 }} align="stretch">
        {/* 设置部分 - 保留所有原有元素 */}
        <Box borderWidth="1px" borderRadius="lg" p={{ base: 3, md: 4 }} bg="whiteAlpha.50">
          <Heading size={{ base: "sm", md: "md" }} mb={{ base: 3, md: 4 }}>{t('Trading Settings')}</Heading>
          
          <SimpleGrid columns={[1, 2, 4]} spacing={{ base: 3, md: 4 }} mb={{ base: 3, md: 4 }}>
            <FormControl size="sm">
              <FormLabel fontSize={{ base: "sm", md: "md" }}>{t('Exchange')}</FormLabel>
              <Select 
                  bg="whiteAlpha.100" 
                  size="sm"
                  sx={{
                  option: {
                    background: "#2D3748 !important",
                    color: "white !important",
                  },
                  "&:not([multiple]) option, &:not([multiple]) optgroup": {
                    backgroundColor: "#2D3748",
                    color: "white"
                  }
                }}
              >
                <option>Binance</option>
                <option>Huobi</option>
                <option>OKX</option>
              </Select>
            </FormControl>
  
            <FormControl size="sm">
              <FormLabel fontSize={{ base: "sm", md: "md" }}>{t('Market')}</FormLabel>
              <Select 
                  bg="whiteAlpha.100" 
                  size="sm"
                  sx={{
                  option: {
                    background: "#2D3748 !important",
                    color: "white !important",
                  },
                  "&:not([multiple]) option, &:not([multiple]) optgroup": {
                    backgroundColor: "#2D3748",
                    color: "white"
                  }
                }}
              >
                <option>Spot</option>
                <option>Futures</option>
              </Select>
            </FormControl>
  
            <FormControl size="sm">
              <FormLabel fontSize={{ base: "sm", md: "md" }}>{t('Total Capital')}</FormLabel>
              <NumberInput size="sm" bg="whiteAlpha.100">
                <NumberInputField />
              </NumberInput>
            </FormControl>
  
            <FormControl size="sm">
              <FormLabel fontSize={{ base: "sm", md: "md" }}>{t('Operation Capital')}</FormLabel>
              <NumberInput size="sm" bg="whiteAlpha.100">
                <NumberInputField />
              </NumberInput>
            </FormControl>
          </SimpleGrid>
  
          <SimpleGrid columns={[2, 2, 3]} spacing={{ base: 3, md: 4 }} mb={{ base: 3, md: 4 }}>
            <FormControl size="sm">
              <FormLabel fontSize={{ base: "sm", md: "md" }}>{t('Stop Loss')}</FormLabel>
              <NumberInput size="sm" bg="whiteAlpha.100">
                <NumberInputField />
              </NumberInput>
            </FormControl>
  
            <FormControl size="sm">
              <FormLabel fontSize={{ base: "sm", md: "md" }}>{t('Take Profit')}</FormLabel>
              <NumberInput size="sm" bg="whiteAlpha.100">
                <NumberInputField />
              </NumberInput>
            </FormControl>
  
            <FormControl size="sm">
              <FormLabel fontSize={{ base: "sm", md: "md" }}>{t('Fee (%)')}</FormLabel>
              <NumberInput size="sm" bg="whiteAlpha.100">
                <NumberInputField />
              </NumberInput>
            </FormControl>
          </SimpleGrid>
  
          <FormControl size="sm" mb={{ base: 3, md: 4 }}>
            <FormLabel fontSize={{ base: "sm", md: "md" }}>{t('Timeframe')}</FormLabel>
            <Select 
              bg="whiteAlpha.100" 
              size="sm"
              sx={{
                option: {
                  background: "#2D3748 !important",
                  color: "white !important",
                },
                "&:not([multiple]) option, &:not([multiple]) optgroup": {
                  backgroundColor: "#2D3748",
                  color: "white"
                }
              }}
            >
              {['5m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d', '3d', '1w'].map(tf => (
                <option key={tf}>{tf}</option>
              ))}
            </Select>
          </FormControl>
  
          <Wrap spacing={{ base: 1, md: 2 }} mt={{ base: 2, md: 4 }}>
            {['RSI', 'MACD', 'KDJ', 'BB', 'MA'].map(indicator => (
              <WrapItem key={indicator}>
                <Button size="sm" variant="outline" colorScheme="blue">
                  {indicator}
                </Button>
              </WrapItem>
            ))}
          </Wrap>
        </Box>
  
        <Box 
          borderWidth="1px" 
          borderRadius="lg" 
          p={{ base: 3, md: 4 }} 
          flex="1"
          minH={{ base: "calc(100vh - 600px)", md: "calc(100vh - 500px)" }}
          bg="whiteAlpha.50"
        >
          <Heading size={{ base: "sm", md: "md" }} mb={{ base: 3, md: 4 }}>{t('Analysis Results')}</Heading>
          
          <Box height={{ base: "300px", md: "400px" }} bg="whiteAlpha.100" borderRadius="lg" mb={{ base: 4, md: 6 }}>
            <Heading size="sm" p={{ base: 3, md: 4 }}>{t('Performance Chart')}</Heading>
          </Box>
  
          <SimpleGrid columns={[2, 2, 4]} spacing={{ base: 3, md: 6 }} mb={{ base: 4, md: 6 }}>
            <Stat bg="whiteAlpha.100" p={{ base: 3, md: 4 }} borderRadius="lg">
              <StatLabel>{t('Win Rate')}</StatLabel>
              <StatNumber color="green.400">68.5%</StatNumber>
              <StatHelpText>
                <StatArrow type="increase" />
                23%
              </StatHelpText>
            </Stat>
  
            <Stat bg="whiteAlpha.100" p={{ base: 3, md: 4 }} borderRadius="lg">
              <StatLabel>{t('Profit Factor')}</StatLabel>
              <StatNumber color="blue.400">2.31</StatNumber>
              <StatHelpText>
                <StatArrow type="increase" />
                0.45
              </StatHelpText>
            </Stat>
  
            <Stat bg="whiteAlpha.100" p={{ base: 3, md: 4 }} borderRadius="lg">
              <StatLabel>{t('Sharpe Ratio')}</StatLabel>
              <StatNumber color="purple.400">2.1</StatNumber>
              <StatHelpText>
                <StatArrow type="increase" />
                0.3
              </StatHelpText>
            </Stat>
  
            <Stat bg="whiteAlpha.100" p={{ base: 3, md: 4 }} borderRadius="lg">
              <StatLabel>{t('Max Drawdown')}</StatLabel>
              <StatNumber color="red.400">12.4%</StatNumber>
              <StatHelpText>
                <StatArrow type="decrease" />
                3%
              </StatHelpText>
            </Stat>
          </SimpleGrid>
  
          <SimpleGrid columns={[1, 1, 2]} spacing={{ base: 4, md: 6 }} mb={{ base: 4, md: 6 }}>
            <Box>
              <Heading size="sm" mb={3}>{t('Current Signals')}</Heading>
              <Stack spacing={3}>
                {['RSI', 'MACD', 'KDJ', 'Overall'].map(signal => (
                  <Stat key={signal} bg="whiteAlpha.100" p={3} borderRadius="md">
                    <StatLabel>{t(`${signal} Signal`)}</StatLabel>
                    <StatNumber fontSize="lg" color="green.400">
                      {t('Buy')}
                    </StatNumber>
                  </Stat>
                ))}
              </Stack>
            </Box>
  
            <Box>
              <Heading size="sm" mb={3}>{t('Additional Metrics')}</Heading>
              <Stack spacing={3}>
                <Stat bg="whiteAlpha.100" p={3} borderRadius="md">
                  <StatLabel>{t('Recovery Factor')}</StatLabel>
                  <StatNumber fontSize="lg">3.2</StatNumber>
                </Stat>
                <Stat bg="whiteAlpha.100" p={3} borderRadius="md">
                  <StatLabel>{t('Sortino Ratio')}</StatLabel>
                  <StatNumber fontSize="lg">2.4</StatNumber>
                </Stat>
                <Stat bg="whiteAlpha.100" p={3} borderRadius="md">
                  <StatLabel>{t('Average Trade')}</StatLabel>
                  <StatNumber fontSize="lg" color="green.400">+1.2%</StatNumber>
                </Stat>
              </Stack>
            </Box>
          </SimpleGrid>
  
          <Box bg="whiteAlpha.100" borderRadius="lg" p={{ base: 2, md: 4 }}>
            <Heading size="sm" mb={{ base: 2, md: 4 }}>{t('Detailed Statistics')}</Heading>
            <Table variant="simple" size="sm">
              <Tbody>
                <Tr>
                  <Td>{t('Total Trades')}</Td>
                  <Td isNumeric>284</Td>
                  <Td>{t('Winning Trades')}</Td>
                  <Td isNumeric>194</Td>
                </Tr>
                <Tr>
                  <Td>{t('Average Win')}</Td>
                  <Td isNumeric>2.3%</Td>
                  <Td>{t('Average Loss')}</Td>
                  <Td isNumeric>-1.1%</Td>
                </Tr>
                <Tr>
                  <Td>{t('Largest Win')}</Td>
                  <Td isNumeric>8.4%</Td>
                  <Td>{t('Largest Loss')}</Td>
                  <Td isNumeric>-3.2%</Td>
                </Tr>
              </Tbody>
            </Table>
          </Box>
        </Box>
  
        <Flex 
          justify="space-between" 
          pt={{ base: 3, md: 4 }}
          flexDir={{ base: "column", md: "row" }}
          gap={{ base: 2, md: 0 }}
        >
          <Button 
            colorScheme="blue" 
            size={{ base: "md", md: "lg" }}
            width={{ base: "100%", md: "auto" }}
          >
            {t('Save Settings')}
          </Button>
          <Button 
            colorScheme="green" 
            size={{ base: "md", md: "lg" }}
            width={{ base: "100%", md: "auto" }}
          >
            {t('Start Trading')}
          </Button>
        </Flex>
      </VStack>
    </Box>
  );

  const PredictionTab = () => {
    const [symbol, setSymbol] = useState('');
    const [loading, setLoading] = useState(false);
    
    return (
      <Box px={{ base: 2, md: 0 }}>  {/* 移动端添加边距 */}
        <VStack spacing={{ base: 4, md: 6 }} align="stretch">
          {/* 搜索区域 */}
          <Card bg="whiteAlpha.100" p={{ base: 4, md: 6 }}>
            <VStack spacing={{ base: 3, md: 4 }}>
              <FormControl>
                <FormLabel 
                  color="gray.300" 
                  fontSize={{ base: "sm", md: "md" }}
                >
                  {t('enterSymbol')}
                </FormLabel>
                <InputGroup size={{ base: "md", md: "lg" }}>
                  <Input
                    placeholder={t('enterSymbolPlaceholder')}
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value)}
                    bg="whiteAlpha.50"
                    color="white"
                    _placeholder={{ color: "gray.500" }}
                    borderColor="whiteAlpha.200"
                    _hover={{ borderColor: "blue.400" }}
                    _focus={{ borderColor: "blue.400", boxShadow: "0 0 0 1px #4299E1" }}
                    fontSize={{ base: "sm", md: "md" }}
                  />
                  <InputRightElement width={{ base: "4rem", md: "4.5rem" }}>
                    <Button 
                      size="sm" 
                      colorScheme="blue" 
                      onClick={() => setLoading(true)}
                      _hover={{ bg: "blue.500" }}
                    >
                      {t('analyze')}
                    </Button>
                  </InputRightElement>
                </InputGroup>
              </FormControl>
            </VStack>
          </Card>
  
          {/* 预测结果区域 */}
          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={{ base: 4, md: 6 }}>
            <Card bg="whiteAlpha.100" p={{ base: 4, md: 6 }}>
              <VStack align="stretch" spacing={{ base: 3, md: 4 }}>
                <Heading 
                  size={{ base: "sm", md: "md" }} 
                  color="gray.100"
                >
                  {t('shortTermPrediction')}
                </Heading>
                <Box>
                  <Text 
                    fontWeight="bold" 
                    color="gray.300"
                    fontSize={{ base: "sm", md: "md" }}
                  >
                    {t('predictionDirection')}
                  </Text>
                  <HStack spacing={{ base: 3, md: 4 }} mt={2}>
                    <Icon 
                      as={FaArrowUp} 
                      color="green.400" 
                      w={{ base: 5, md: 6 }} 
                      h={{ base: 5, md: 6 }}
                    />
                    <Text 
                      color="green.400" 
                      fontSize={{ base: "lg", md: "xl" }} 
                      fontWeight="bold"
                    >
                      +2.3%
                    </Text>
                  </HStack>
                </Box>
                <Text 
                  fontSize={{ base: "xs", md: "sm" }} 
                  color="gray.500"
                >
                  {t('nextHours')}
                </Text>
              </VStack>
            </Card>
  
            <Card bg="whiteAlpha.100" p={{ base: 4, md: 6 }}>
              <VStack align="stretch" spacing={{ base: 3, md: 4 }}>
                <Heading 
                  size={{ base: "sm", md: "md" }} 
                  color="gray.100"
                >
                  {t('technicalIndicators')}
                </Heading>
                <SimpleGrid 
                  columns={2} 
                  spacing={{ base: 3, md: 4 }}
                  fontSize={{ base: "sm", md: "md" }}
                >
                  <Box>
                    <Text fontWeight="bold" color="gray.300">RSI</Text>
                    <Text color="white">63.5</Text>
                  </Box>
                  <Box>
                    <Text fontWeight="bold" color="gray.300">MACD</Text>
                    <Text color="green.400">Bullish</Text>
                  </Box>
                  <Box>
                    <Text fontWeight="bold" color="gray.300">MA</Text>
                    <Text color="green.400">Above 200</Text>
                  </Box>
                  <Box>
                    <Text fontWeight="bold" color="gray.300">Volume</Text>
                    <Text color="blue.300">+15%</Text>
                  </Box>
                </SimpleGrid>
              </VStack>
            </Card>
          </SimpleGrid>
  
          {/* 渐变分割线 */}
          <Box
            h="1px"
            bg="linear-gradient(90deg, transparent, whiteAlpha.300, transparent)"
            my={{ base: 3, md: 4 }}
          />
        </VStack>
      </Box>
    );
  };

  const PortfolioTab = () => {
    return (
      <Box>
        <VStack spacing={6} align="stretch">
          {/* 投资组合概览 */}
          <Card bg="whiteAlpha.100" p={6}>
            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
              <Stat>
                <StatLabel color="gray.300">{t('totalValue')}</StatLabel>
                <StatNumber color="white" fontSize="2xl">$85,000</StatNumber>
                <StatHelpText color="green.400">
                  <StatArrow type="increase" />
                  8.2%
                </StatHelpText>
              </Stat>
              <Stat>
                <StatLabel color="gray.300">{t('riskScore')}</StatLabel>
                <StatNumber color="orange.400" fontSize="2xl">7.2/10</StatNumber>
                <StatHelpText color="gray.400">{t('moderateRisk')}</StatHelpText>
              </Stat>
              <Stat>
                <StatLabel color="gray.300">{t('diversification')}</StatLabel>
                <StatNumber color="blue.400" fontSize="2xl">8.5/10</StatNumber>
                <StatHelpText color="blue.400">{t('wellDiversified')}</StatHelpText>
              </Stat>
            </SimpleGrid>
          </Card>
  
          {/* 资产分配和建议 */}
          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
            <Card bg="whiteAlpha.100" p={6}>
              <VStack align="stretch" spacing={4}>
                <Heading size="md" color="gray.100">{t('currentAllocation')}</Heading>
                <Box h="200px" position="relative">
                  {/* 这里可以添加饼图 */}
                  <Text color="gray.400">{t('assetDistribution')}</Text>
                </Box>
              </VStack>
            </Card>
  
            <Card bg="whiteAlpha.100" p={6}>
              <VStack align="stretch" spacing={4}>
                <Heading size="md" color="gray.100">{t('rebalancingSuggestions')}</Heading>
                <List spacing={3}>
                  {[
                    { action: 'Increase', asset: 'BTC', change: '+5%' },
                    { action: 'Decrease', asset: 'AAPL', change: '-3%' },
                    { action: 'Hold', asset: 'ETH', change: '0%' },
                  ].map((item, index) => (
                    <ListItem key={index} color="white">
                      <HStack>
                        <Icon 
                          as={item.action === 'Increase' ? FaArrowUp : item.action === 'Decrease' ? FaArrowDown : FaMinus}
                          color={item.action === 'Increase' ? 'green.400' : item.action === 'Decrease' ? 'red.400' : 'gray.400'}
                        />
                        <Text>{`${item.asset}: ${item.change}`}</Text>
                      </HStack>
                    </ListItem>
                  ))}
                </List>
              </VStack>
            </Card>
          </SimpleGrid>
  
          {/* 风险分析 */}
          <Card bg="whiteAlpha.100" p={6}>
            <VStack align="stretch" spacing={4}>
              <Heading size="md" color="gray.100">{t('riskMetrics')}</Heading>
              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                <Box>
                  <Text color="gray.300" mb={2}>{t('volatility')}</Text>
                  <Progress value={65} colorScheme="orange" borderRadius="full" />
                </Box>
                <Box>
                  <Text color="gray.300" mb={2}>{t('sharpRatio')}</Text>
                  <Progress value={80} colorScheme="green" borderRadius="full" />
                </Box>
              </SimpleGrid>
            </VStack>
          </Card>
        </VStack>
      </Box>
    );
  };

  const RiskAnalysisTab = () => {
    return (
      <Box px={{ base: 2, md: 0 }}>  {/* 移动端添加边距 */}
        <VStack spacing={{ base: 4, md: 6 }} align="stretch">
          {/* 风险概览 */}
          <Card bg="whiteAlpha.100" p={{ base: 4, md: 6 }}>
            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={{ base: 4, md: 6 }}>
              <Stat>
                <StatLabel 
                  color="gray.300" 
                  fontSize={{ base: "sm", md: "md" }}
                >
                  {t('overallRisk')}
                </StatLabel>
                <StatNumber 
                  color="purple.400" 
                  fontSize={{ base: "xl", md: "2xl" }}
                >
                  6.8/10
                </StatNumber>
                <StatHelpText 
                  color="gray.400"
                  fontSize={{ base: "xs", md: "sm" }}
                >
                  {t('moderateRisk')}
                </StatHelpText>
              </Stat>
              <Stat>
                <StatLabel 
                  color="gray.300"
                  fontSize={{ base: "sm", md: "md" }}
                >
                  {t('riskTrend')}
                </StatLabel>
                <StatNumber 
                  color="green.400" 
                  fontSize={{ base: "xl", md: "2xl" }}
                >
                  <StatArrow type="decrease" />
                  12%
                </StatNumber>
                <StatHelpText 
                  color="green.400"
                  fontSize={{ base: "xs", md: "sm" }}
                >
                  {t('decreasing')}
                </StatHelpText>
              </Stat>
              <Stat>
                <StatLabel 
                  color="gray.300"
                  fontSize={{ base: "sm", md: "md" }}
                >
                  {t('alertLevel')}
                </StatLabel>
                <StatNumber 
                  color="yellow.400" 
                  fontSize={{ base: "xl", md: "2xl" }}
                >
                  {t('moderate')}
                </StatNumber>
                <StatHelpText 
                  color="yellow.400"
                  fontSize={{ base: "xs", md: "sm" }}
                >
                  {t('needsAttention')}
                </StatHelpText>
              </Stat>
            </SimpleGrid>
          </Card>
  
          {/* 风险指标详情 */}
          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={{ base: 4, md: 6 }}>
            <Card bg="whiteAlpha.100" p={{ base: 4, md: 6 }}>
              <VStack align="stretch" spacing={{ base: 3, md: 4 }}>
                <Heading 
                  size={{ base: "sm", md: "md" }} 
                  color="gray.100"
                >
                  {t('keyRiskMetrics')}
                </Heading>
                <VStack align="stretch" spacing={{ base: 2, md: 3 }}>
                  <Box>
                    <Text 
                      color="gray.300" 
                      mb={2}
                      fontSize={{ base: "sm", md: "md" }}
                    >
                      {t('volatilityIndex')}
                    </Text>
                    <Progress 
                      value={68} 
                      colorScheme="purple" 
                      borderRadius="full" 
                      size={{ base: "sm", md: "md" }}
                    />
                  </Box>
                  <Box>
                    <Text 
                      color="gray.300" 
                      mb={2}
                      fontSize={{ base: "sm", md: "md" }}
                    >
                      {t('betaValue')}
                    </Text>
                    <Progress 
                      value={75} 
                      colorScheme="blue" 
                      borderRadius="full"
                      size={{ base: "sm", md: "md" }}
                    />
                  </Box>
                  <Box>
                    <Text 
                      color="gray.300" 
                      mb={2}
                      fontSize={{ base: "sm", md: "md" }}
                    >
                      {t('drawdownRisk')}
                    </Text>
                    <Progress 
                      value={45} 
                      colorScheme="orange" 
                      borderRadius="full"
                      size={{ base: "sm", md: "md" }}
                    />
                  </Box>
                </VStack>
              </VStack>
            </Card>
  
            <Card bg="whiteAlpha.100" p={{ base: 4, md: 6 }}>
              <VStack align="stretch" spacing={{ base: 3, md: 4 }}>
                <Heading 
                  size={{ base: "sm", md: "md" }} 
                  color="gray.100"
                >
                  {t('riskDistribution')}
                </Heading>
                <Box h={{ base: "150px", md: "200px" }} position="relative">
                  {/* 这里可以添加风险分布图表 */}
                </Box>
              </VStack>
            </Card>
          </SimpleGrid>
  
          {/* 风险管理建议 */}
          <Card bg="whiteAlpha.100" p={{ base: 4, md: 6 }}>
            <VStack align="stretch" spacing={{ base: 3, md: 4 }}>
              <Heading 
                size={{ base: "sm", md: "md" }} 
                color="gray.100"
              >
                {t('riskManagementSuggestions')}
              </Heading>
              <List spacing={{ base: 2, md: 3 }}>
                {[
                  { priority: 'High', message: t('reduceExposure'), impact: 'Critical' },
                  { priority: 'Medium', message: t('diversifyPortfolio'), impact: 'Moderate' },
                  { priority: 'Low', message: t('hedgePositions'), impact: 'Low' }
                ].map((item, index) => (
                  <ListItem key={index}>
                    <HStack 
                      spacing={{ base: 2, md: 4 }}
                      fontSize={{ base: "sm", md: "md" }}
                      flexWrap={{ base: "wrap", md: "nowrap" }}
                    >
                      <Tag 
                        size={{ base: "sm", md: "md" }}
                        colorScheme={
                          item.priority === 'High' ? 'red' : 
                          item.priority === 'Medium' ? 'yellow' : 'green'
                        }
                      >
                        {item.priority}
                      </Tag>
                      <Text color="white">{item.message}</Text>
                      <Tag 
                        size={{ base: "sm", md: "md" }} 
                        variant="subtle"
                      >
                        {item.impact}
                      </Tag>
                    </HStack>
                  </ListItem>
                ))}
              </List>
            </VStack>
          </Card>
        </VStack>
      </Box>
    );
  };

  const AlertsTab = () => {
    const [activeAlerts, setActiveAlerts] = useState(true);
  
    return (
      <Box>
        <VStack spacing={6} align="stretch">
          {/* 预警概览 */}
          <Card bg="whiteAlpha.100" p={6}>
            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
              <Stat>
                <StatLabel color="gray.300">{t('activeAlerts')}</StatLabel>
                <StatNumber color="yellow.400" fontSize="2xl">12</StatNumber>
                <StatHelpText color="gray.400">
                  <StatArrow type="increase" />
                  {t('from3Days')}
                </StatHelpText>
              </Stat>
              <Stat>
                <StatLabel color="gray.300">{t('alertAccuracy')}</StatLabel>
                <StatNumber color="green.400" fontSize="2xl">92%</StatNumber>
                <StatHelpText color="green.400">{t('lastMonth')}</StatHelpText>
              </Stat>
              <Stat>
                <StatLabel color="gray.300">{t('priorityAlerts')}</StatLabel>
                <StatNumber color="red.400" fontSize="2xl">3</StatNumber>
                <StatHelpText color="red.400">{t('needsAction')}</StatHelpText>
              </Stat>
            </SimpleGrid>
          </Card>
  
          {/* 预警设置和列表 */}
          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
            {/* 预警设置 */}
            <Card bg="whiteAlpha.100" p={6}>
              <VStack align="stretch" spacing={4}>
                <Heading size="md" color="gray.100">{t('alertSettings')}</Heading>
                <FormControl>
                  <FormLabel color="gray.300">{t('alertsEnabled')}</FormLabel>
                  <Switch colorScheme="yellow" isChecked={activeAlerts} onChange={(e) => setActiveAlerts(e.target.checked)} />
                </FormControl>
                <FormControl>
                  <FormLabel color="gray.300">{t('priceThreshold')}</FormLabel>
                  <Slider defaultValue={5} min={1} max={10} step={1} colorScheme="yellow">
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </FormControl>
                <FormControl>
                  <FormLabel color="gray.300">{t('notificationChannels')}</FormLabel>
                  <CheckboxGroup colorScheme="yellow" defaultValue={['email', 'push']}>
                    <VStack align="start">
                      <Checkbox value="email" color="gray.300">{t('email')}</Checkbox>
                      <Checkbox value="push" color="gray.300">{t('push')}</Checkbox>
                      <Checkbox value="sms" color="gray.300">{t('sms')}</Checkbox>
                    </VStack>
                  </CheckboxGroup>
                </FormControl>
              </VStack>
            </Card>
  
            {/* 活跃预警 */}
            <Card bg="whiteAlpha.100" p={6}>
              <VStack align="stretch" spacing={4}>
                <Heading size="md" color="gray.100">{t('activeAlertsList')}</Heading>
                <List spacing={3}>
                  {[
                    { type: 'price', asset: 'BTC', message: t('priceAboveThreshold'), priority: 'high' },
                    { type: 'volume', asset: 'ETH', message: t('unusualVolume'), priority: 'medium' },
                    { type: 'pattern', asset: 'AAPL', message: t('bullishPattern'), priority: 'low' }
                  ].map((alert, index) => (
                    <ListItem key={index}>
                      <HStack spacing={4}>
                        <Icon 
                          as={
                            alert.type === 'price' ? FaChartLine :
                            alert.type === 'volume' ? FaChartBar : FaChartArea
                          }
                          color={
                            alert.priority === 'high' ? 'red.400' :
                            alert.priority === 'medium' ? 'yellow.400' : 'green.400'
                          }
                        />
                        <VStack align="start" spacing={0}>
                          <Text color="white" fontWeight="bold">{alert.asset}</Text>
                          <Text color="gray.400" fontSize="sm">{alert.message}</Text>
                        </VStack>
                        <Spacer />
                        <IconButton
                          aria-label="Dismiss alert"
                          icon={<FaTimes />}
                          size="sm"
                          variant="ghost"
                          colorScheme="whiteAlpha"
                        />
                      </HStack>
                    </ListItem>
                  ))}
                </List>
              </VStack>
            </Card>
          </SimpleGrid>
  
          {/* 历史预警 */}
          <Card bg="whiteAlpha.100" p={6}>
            <VStack align="stretch" spacing={4}>
              <Heading size="md" color="gray.100">{t('alertHistory')}</Heading>
              <Table variant="simple" size="sm">
                <Thead>
                  <Tr>
                    <Th color="gray.400">{t('date')}</Th>
                    <Th color="gray.400">{t('asset')}</Th>
                    <Th color="gray.400">{t('type')}</Th>
                    <Th color="gray.400">{t('result')}</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {[
                    { date: '2024-10-24', asset: 'BTC', type: t('priceAlert'), result: t('successful') },
                    { date: '2024-10-23', asset: 'ETH', type: t('volumeAlert'), result: t('successful') },
                    { date: '2024-10-22', asset: 'AAPL', type: t('patternAlert'), result: t('missed') }
                  ].map((history, index) => (
                    <Tr key={index}>
                      <Td color="gray.300">{history.date}</Td>
                      <Td color="gray.300">{history.asset}</Td>
                      <Td color="gray.300">{history.type}</Td>
                      <Td>
                        <Tag
                          size="sm"
                          colorScheme={history.result === t('successful') ? 'green' : 'red'}
                        >
                          {history.result}
                        </Tag>
                      </Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </VStack>
          </Card>
        </VStack>
      </Box>
    );
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
        <Button colorScheme="red" size="sm" onClick={handleLogout}>
          {t('Log out')}
        </Button>
      </Flex>
  
      <Heading as="h1" size="2xl" textAlign="center" mb={6}>
        {t('AI Trading Dashboard')}
      </Heading>
  
      <Tabs variant="soft-rounded" colorScheme="blue" mt={4}  index={tabIndex} 
      onChange={setTabIndex}>
        <TabList mb={4}>
          <Tab>{t('Dashboard')}</Tab>
          <Tab>{t('Recall')}</Tab>
          <Tab>{t('prediction')}</Tab>
          <Tab>{t('portfolio')}</Tab>
          <Tab>{t('risk')}</Tab> 
          <Tab>{t('alerts')}</Tab>
        </TabList>
  
        <TabPanels>
          <TabPanel>
            {/* Dashboard面板添加滚动 */}
            <Box 
              height="calc(100vh - 250px)" 
              overflowY="auto" 
              pr={2}
              css={{
                '&::-webkit-scrollbar': {
                  width: '4px',
                },
                '&::-webkit-scrollbar-track': {
                  width: '6px',
                },
                '&::-webkit-scrollbar-thumb': {
                  background: 'gray.500',
                  borderRadius: '24px',
                },
              }}
            >
              {renderDashboard()}
            </Box>
          </TabPanel>
          
          <TabPanel>
            {/* Settings面板已经有滚动设置，保持不变 */}
            {renderSettings()}
          </TabPanel>
          <TabPanel>
      <PredictionTab />
    </TabPanel>
    <TabPanel>
      <PortfolioTab />  {/* 新增的内容 */}
    </TabPanel>
    <TabPanel>
      <RiskAnalysisTab />  {/* 新增的内容 */}
    </TabPanel>
    <TabPanel>
      <AlertsTab />  {/* 新增的内容 */}
    </TabPanel>
        </TabPanels>
      </Tabs>  
    </Box>
  );
}
