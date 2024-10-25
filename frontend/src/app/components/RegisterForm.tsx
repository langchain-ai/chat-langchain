// In RegisterForm.tsx
import React from "react";
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  VStack,
  Heading,
  Text,
} from "@chakra-ui/react";
import { useTranslations, useLocale } from 'next-intl';
import { useRouter, usePathname, Link } from '@/src/i18n/routing';

interface RegisterFormProps {
  onBackToLogin: () => void;  // 添加这个属性
}

export function RegisterForm({ onBackToLogin }: RegisterFormProps) {
  const router = useRouter();
  const pathname = usePathname();
  const locale = useLocale();  // 获取当前语言

  const handleLocaleChange = (newLocale: string) => {
      // router.replace(`/${newLocale}${pathname}`); // 确保切换时 URL 正确
      window.location.href =  `/${newLocale}${pathname}`;
    };
 const t = useTranslations('HomePage'); 
  return (
    <Box p={8} bg="#1A202C" color="white" height="100%">
      <VStack spacing={6} align="center" justify="center" height="100%">
        <Heading as="h1" size="2xl" textAlign="center" mb={8}>
        {t('Register')}      
        </Heading>
        <Box width="100%" maxWidth="400px">
          <FormControl id="email" mb={4}>
            <FormLabel>{t('Email address')} </FormLabel>
            <Input type="email" placeholder={t('Enter your email')} />
          </FormControl>
          <FormControl id="password" mb={4}>
            <FormLabel>{t('Password')}</FormLabel>
            <Input type="password" placeholder={t('Enter your password')} />
          </FormControl>
          <FormControl id="confirmPassword" mb={8}>
            <FormLabel>{t('Confirm Password')}</FormLabel>
            <Input type="password" placeholder={t('Confirm your password')} />
          </FormControl>
          <Button
            colorScheme="blue"
            size="lg"
            width="100%"
            mb={4}
            bg="#3182CE"
            _hover={{ bg: "#2B6CB0" }}
          >
            {t('Register')} 
          </Button>
          <Text fontSize="sm" textAlign="center">
          {t('Already have an account?')} {" "}
            <Text as="a" color="blue.400" onClick={onBackToLogin} cursor="pointer">
            {t('Login')}
            </Text>
          </Text>
        </Box>
      </VStack>
    </Box>
  );
}
