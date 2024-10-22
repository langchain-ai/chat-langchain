// In PricingPlan.tsx
import React from "react";
import { Box, Heading, Text, VStack, Image, Flex } from "@chakra-ui/react";
import { useTranslations, useLocale } from 'next-intl';
import { useRouter, usePathname, Link } from '@/src/i18n/routing';

export function PricingPlan() {
  const router = useRouter();
  const pathname = usePathname();
  const locale = useLocale();  // 获取当前语言

  const handleLocaleChange = (newLocale: string) => {
      // router.replace(`/${newLocale}${pathname}`); // 确保切换时 URL 正确
      window.location.href =  `/${newLocale}${pathname}`;
    };
  const t = useTranslations('HomePage'); 
  return (
    <Box p={8} bg="#1A202C" color="white">
      <VStack spacing={8} align="stretch">
        <Heading as="h1" size="2xl" textAlign="center">
          {t('RichMaster StockGPT PricingPlan')}
        </Heading>
        <Text fontSize="xl" textAlign="center">
        {t('Coming Soon！')} 
        </Text>
      </VStack>
    </Box>
  );
}
