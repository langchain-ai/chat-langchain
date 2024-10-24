import { useState } from 'react';
import { useToast } from '@chakra-ui/react';
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

interface LoginProps {
  onLoginSuccess: () => void;
  
}

export default function Login({ onLoginSuccess }: LoginProps) {
    
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
    <Box p={8} bg="#1A202C" color="white" height="100%">
      <VStack spacing={6} align="center" justify="center" height="100%">
        <Heading as="h1" size="2xl" textAlign="center" mb={8}>
          {t('Login')}
        </Heading>
        <Box width="100%" maxWidth="400px">
          <form onSubmit={handleSubmit}>
            <FormControl id="email" mb={4}>
              <FormLabel>{t('Email address')}</FormLabel>
              <Input
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                placeholder={t('Enter your email')}
                required
              />
            </FormControl>
            
            <FormControl id="password" mb={8}>
              <FormLabel>{t('Password')}</FormLabel>
              <Input
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                placeholder={t('Enter your password')}
                required
              />
            </FormControl>

            <Button
              type="submit"
              colorScheme="blue"
              size="lg"
              width="100%"
              mb={4}
              bg="#3182CE"
              _hover={{ bg: "#2B6CB0" }}
              isLoading={loading}
              loadingText={t('Logging in')}
            >
              {t('Login')}
            </Button>
          </form>

          <Text fontSize="sm" textAlign="center">
            {t("Don't have an account?")} {" "}
            <Text as="span" color="blue.400" cursor="pointer">
              {t('Register')}
            </Text>
          </Text>
        </Box>
      </VStack>
    </Box>
  );
}