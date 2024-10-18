// In PricingPlan.tsx
import React from "react";
import { Box, Heading, Text, VStack, Image, Flex } from "@chakra-ui/react";

export function PricingPlan() {
  return (
    <Box p={8} bg="#1A202C" color="white">
      <VStack spacing={8} align="stretch">
        <Heading as="h1" size="2xl" textAlign="center">
          RichMaster StockGPT PricingPlan
        </Heading>
        <Text fontSize="xl" textAlign="center">
          Coming Soon！
        </Text>
      </VStack>
    </Box>
  );
}
