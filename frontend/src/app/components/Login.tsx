'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  VStack,
  Container,
  Heading,
  useToast,
} from '@chakra-ui/react';

export default function LoginPage() {
  const router = useRouter();
  const toast = useToast();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });

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
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (data.success) {
        // 设置 cookie
        document.cookie = `token=${data.token}; path=/`;
        
        toast({
          title: '登录成功',
          status: 'success',
          duration: 2000,
        });

        // 重定向到 rich-master-ai 页面
        router.push('/rich-master-ai');
      } else {
        throw new Error(data.message);
      }
    } catch (error: any) {
      toast({
        title: '登录失败',
        description: error.message || '请检查邮箱和密码',
        status: 'error',
        duration: 3000,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxW="md" py={12}>
      <VStack spacing={8}>
        <Heading>登录</Heading>
        <Box as="form" onSubmit={handleSubmit} width="100%">
          <VStack spacing={4}>
            <FormControl isRequired>
              <FormLabel>邮箱</FormLabel>
              <Input
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="请输入邮箱"
              />
            </FormControl>

            <FormControl isRequired>
              <FormLabel>密码</FormLabel>
              <Input
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="请输入密码"
              />
            </FormControl>

            <Button
              type="submit"
              colorScheme="blue"
              width="100%"
              isLoading={loading}
              loadingText="登录中..."
            >
              登录
            </Button>
          </VStack>
        </Box>
      </VStack>
    </Container>
  );
}