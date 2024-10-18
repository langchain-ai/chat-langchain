// In AboutUs.tsx
import React from "react";
import { Box, Heading, Text, VStack, Image, Flex } from "@chakra-ui/react";

export function AboutUs() {
  return (
    <Box p={8} bg="#1A202C" color="white">
      <VStack spacing={8} align="stretch">
        <Heading as="h1" size="2xl" textAlign="center">
          About RichMaster StockGPT
        </Heading>
        <Text fontSize="xl" textAlign="center">
          We are a team of passionate individuals dedicated to providing the best stock trading experience through cutting-edge AI technology.
        </Text>
        <Box>
          <Heading as="h2" size="xl" mb={4}>
            Our Mission
          </Heading>
          <Text fontSize="lg">
            Our mission is to empower traders with intelligent tools and insights to make informed decisions in the stock market. We strive to simplify complex financial data and provide easy-to-use interfaces for traders of all levels.
          </Text>
        </Box>
      </VStack>
    </Box>
  );
}
