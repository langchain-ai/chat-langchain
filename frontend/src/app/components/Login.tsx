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
      p={{ base: 4, md: 8 }}  // 调整响应式padding
      bg="#1A202C" 
      color="white" 
      minHeight="100vh"
      bgGradient="linear(to-br, #1A202C, #2D3748)"
    >
      <SimpleGrid 
        columns={{ base: 1, md: 2 }} 
        spacing={{ base: 6, md: 10 }}
        maxW="1400px"  // 增加最大宽度
        mx="auto"
        minHeight="80vh"  // 设置最小高度
        alignItems="stretch" // 让子元素撑满高度
      >
        {/* 左侧登录表单 */}
        <VStack 
          spacing={8} 
          align="center" 
          justify="center"
          bg="whiteAlpha.50"
          p={{ base: 6, md: 12 }}  // 增加内边距
          borderRadius="xl"
          boxShadow="xl"
          minHeight="600px"  // 设置最小高度
        >
          <Heading 
            as="h1" 
            size="2xl" 
            textAlign="center" 
            mb={8}
            bgGradient="linear(to-r, blue.400, blue.600)"
            bgClip="text"
          >
            {t('RichMaster StockGPT')}
          </Heading>
          <Box width="100%" maxWidth="450px">  
            <form onSubmit={handleSubmit}>
              <FormControl id="email" mb={6}>  
                <FormLabel fontSize="lg">{t('Email address')}</FormLabel>
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
                  size="lg"  // 增大输入框
                  height="56px"  // 自定义高度
                />
              </FormControl>
              
              <FormControl id="password" mb={8}>
                <FormLabel fontSize="lg">{t('Password')}</FormLabel>
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
                  size="lg"
                  height="56px"
                />
              </FormControl>
  
              <Button
                type="submit"
                colorScheme="blue"
                size="lg"
                width="100%"
                mb={6}
                bg="#3182CE"
                _hover={{ bg: "#2B6CB0" }}
                isLoading={loading}
                loadingText={t('Logging in')}
                height="56px"  // 统一按钮高度
                fontSize="lg"  // 增大字体
              >
                {t('Login')}
              </Button>
            </form>
  
            <Text fontSize="md" textAlign="center">
              {t("Don't have an account?")} {" "}
              <Text 
                as="span" 
                color="blue.400" 
                cursor="pointer" 
                onClick={onRegisterClick}
                _hover={{ color: "blue.300" }}
              >
                {t('Register')}
              </Text>
            </Text>
          </Box>
        </VStack>
  
        {/* 右侧产品介绍 */}
        <VStack 
          spacing={8} 
          align="flex-start" 
          justify="center"
          display={{ base: 'none', md: 'flex' }}
          p={{ base: 6, md: 12 }}
          bg="whiteAlpha.50"
          borderRadius="xl"
          boxShadow="xl"
          minHeight="600px"  // 保持与左侧相同高度
        >
          <VStack align="flex-start" spacing={8} width="100%">
            <Heading 
              size="xl"  // 增大标题
              bgGradient="linear(to-r, blue.400, teal.400)"
              bgClip="text"
            >
              {t('tradingPlatform')}
            </Heading>
            
            <Text fontSize="xl" lineHeight="tall">  
              {t('platformDesc1')}
            </Text>
  
            <Text fontSize="xl" lineHeight="tall">
              {t('platformDesc2')}
            </Text>
  
            {/* 特性标签 */}
            <Wrap spacing={4} mt={4}>
              {[
                '技术分析', 'AI预测', '历史回溯', '实时数据'
              ].map((tag) => (
                <WrapItem key={tag}>
                  <Tag 
                    size="lg" 
                    colorScheme="blue" 
                    variant="subtle"
                    py={2}  // 增加标签高度
                    px={4}  // 增加标签宽度
                    fontSize="md"  // 增大标签字体
                  >
                    {tag}
                  </Tag>
                </WrapItem>
              ))}
            </Wrap>
  
            {/* 图标说明 */}
            <SimpleGrid 
              columns={2} 
              spacing={8}  // 增加图标间距
              width="100%"
              mt={4}
            >
              <HStack spacing={4}>  
                <Icon as={FaChartLine} w={8} h={8} color="blue.400" />  
                <Text fontSize="lg">{t('technicalAnalysis')}</Text>
              </HStack>
              <HStack spacing={4}>
                <Icon as={FaBrain} w={8} h={8} color="teal.400" />
                <Text fontSize="lg">{t('aiPrediction')}</Text>
              </HStack>
            </SimpleGrid>
          </VStack>
        </VStack>
      </SimpleGrid>
    </Box>
  );
}