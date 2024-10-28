import { useState } from 'react';
import { HStack, Icon, SimpleGrid, Tag, useToast, Wrap, WrapItem } from '@chakra-ui/react';
import {
    Box,
    VStack,
    Heading,
    FormControl,
    FormLabel,
    Input,
    Button,
    Text
  } from '@chakra-ui/react';
  import { useTranslations } from 'next-intl';
import { FaBrain, FaChartLine } from 'react-icons/fa';

interface LoginProps {
  onLoginSuccess: () => void;
  onRegisterClick: () => void;
}

export default function Login({ onLoginSuccess, onRegisterClick }: LoginProps) {
    
  const t = useTranslations('HomePage'); 
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const toast = useToast();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
  
    try {
      console.log('Sending login request:', formData); // 添加调试日志
      const response = await fetch('/api/auth/login', { // 修改这里的路径
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password
          // 移除 action: 'login'
        }),
      });

      const data = await response.json();

      if (data.success) {
        // 设置 cookie
        document.cookie = `token=${data.token}; path=/`;
        onLoginSuccess(); // 通知父组件登录成功
        toast({
          title: "登录成功",
          status: "success",
          duration: 3000,
        });
      } else {
        throw new Error(data.message);
      }
    } catch (error: any) {
      toast({
        title: "登录失败",
        description: error.message || "请检查用户名和密码",
        status: "error",
        duration: 3000,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box 
      p={{ base: 2, md: 8 }}
      bg="#1A202C" 
      color="white" 
      minHeight="100vh"
      bgGradient="linear(to-br, #1A202C, #2D3748)"
    >
      <SimpleGrid 
        columns={{ base: 1, md: 2 }} 
        spacing={{ base: 4, md: 10 }}
        maxW="1400px"
        mx="auto"
        minHeight={{ base: "auto", md: "80vh" }}
        alignItems="stretch"
      >
        {/* 左侧产品介绍 (桌面端) / 顶部产品介绍 (移动端) */}
        <VStack 
          spacing={{ base: 4, md: 8 }}
          align="flex-start" 
          justify="center"
          display={{ base: 'flex', md: 'flex' }}
          p={{ base: 4, md: 12 }}
          bg="whiteAlpha.50"
          borderRadius="xl"
          boxShadow="xl"
          minHeight={{ base: "auto", md: "600px" }}
          order={{ base: 1, md: 1 }} // 确保移动端在顶部，桌面端在左侧
        >
          <VStack align="flex-start" spacing={{ base: 4, md: 8 }} width="100%">
            <Heading 
              size={{ base: "lg", md: "xl" }}
              bgGradient="linear(to-r, blue.400, teal.400)"
              bgClip="text"
            >
              {t('tradingPlatform')}
            </Heading>
            
            <Text fontSize={{ base: "md", md: "xl" }} lineHeight="tall">  
              {t('platformDesc1')}
            </Text>
  
            <Text 
              fontSize={{ base: "md", md: "xl" }} 
              lineHeight="tall"
              display={{ base: "none", md: "block" }}
            >
              {t('platformDesc2')}
            </Text>
  
            <Wrap spacing={{ base: 2, md: 4 }} mt={{ base: 2, md: 4 }}>
              {[
                 t('Technical Analysis'), t('AI Prediction'), t('Realtime Data')
              ].map((tag) => (
                <WrapItem key={tag}>
                  <Tag 
                    size={{ base: "md", md: "lg" }}
                    colorScheme="blue" 
                    variant="subtle"
                    py={{ base: 1, md: 2 }}
                    px={{ base: 3, md: 4 }}
                    fontSize={{ base: "sm", md: "md" }}
                  >
                    {tag}
                  </Tag>
                </WrapItem>
              ))}
            </Wrap>
  
            <SimpleGrid 
              columns={{ base: 1, md: 2 }}
              spacing={{ base: 4, md: 8 }}
              width="100%"
              mt={{ base: 2, md: 4 }}
            >
              <HStack spacing={{ base: 3, md: 4 }}>  
                <Icon as={FaChartLine} w={{ base: 6, md: 8 }} h={{ base: 6, md: 8 }} color="blue.400" />  
                <Text fontSize={{ base: "md", md: "lg" }}>{t('technicalAnalysis')}</Text>
              </HStack>
              <HStack spacing={{ base: 3, md: 4 }}>
                <Icon as={FaBrain} w={{ base: 6, md: 8 }} h={{ base: 6, md: 8 }} color="teal.400" />
                <Text fontSize={{ base: "md", md: "lg" }}>{t('aiPrediction')}</Text>
              </HStack>
            </SimpleGrid>
          </VStack>
        </VStack>
  
        {/* 右侧登录表单 (桌面端) / 底部登录表单 (移动端) */}
        <VStack 
          spacing={{ base: 4, md: 8 }}
          align="center" 
          justify="center"
          bg="whiteAlpha.50"
          p={{ base: 4, md: 12 }}
          borderRadius="xl"
          boxShadow="xl"
          minHeight={{ base: "auto", md: "600px" }}
          order={{ base: 2, md: 2 }} // 确保移动端在底部，桌面端在右侧
        >
          <Heading 
            as="h1" 
            size={{ base: "xl", md: "2xl" }}
            textAlign="center" 
            mb={{ base: 4, md: 8 }}
            bgGradient="linear(to-r, blue.400, blue.600)"
            bgClip="text"
          >
            {t('RichMaster StockGPT')}
          </Heading>
          <Box width="100%" maxWidth="450px">  
            <form onSubmit={handleSubmit}>
              <FormControl id="email" mb={{ base: 4, md: 6 }}>  
                <FormLabel fontSize={{ base: "md", md: "lg" }}>{t('Email address')}</FormLabel>
                <Input
                  name="email"
                  type="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder={t('Enter your email')}
                  required
                  bg="whiteAlpha.100"
                  borderColor="whiteAlpha.300"
                  _hover={{ borderColor: "blue.400" }}
                  size={{ base: "md", md: "lg" }}
                  height={{ base: "48px", md: "56px" }}
                />
              </FormControl>
              
              <FormControl id="password" mb={{ base: 6, md: 8 }}>
                <FormLabel fontSize={{ base: "md", md: "lg" }}>{t('Password')}</FormLabel>
                <Input
                  name="password"
                  type="password"
                  value={formData.password}
                  onChange={handleChange}
                  placeholder={t('Enter your password')}
                  required
                  bg="whiteAlpha.100"
                  borderColor="whiteAlpha.300"
                  _hover={{ borderColor: "blue.400" }}
                  size={{ base: "md", md: "lg" }}
                  height={{ base: "48px", md: "56px" }}
                />
              </FormControl>
  
              <Button
                type="submit"
                colorScheme="blue"
                isLoading={loading}
                loadingText={t('Logging in')}
                _hover={{ bg: "#2B6CB0" }}
                bg="#3182CE"
                size={{ base: "md", md: "lg" }}
                height={{ base: "48px", md: "56px" }}
                fontSize={{ base: "md", md: "lg" }}
                mb={{ base: 4, md: 6 }}
                width="100%"
              >
                {t('Login')}
              </Button>
            </form>
  
            <Text fontSize={{ base: "sm", md: "md" }} textAlign="center">
              {/* ... 保持不变 */}
            </Text>
          </Box>
        </VStack>
      </SimpleGrid>
    </Box>
  );
}
