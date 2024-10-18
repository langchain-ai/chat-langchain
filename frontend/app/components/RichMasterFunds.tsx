// In RichMasterFunds.tsx
import React from "react";
import { Box, Heading, Text, VStack, Image, Flex } from "@chakra-ui/react";

export function RichMasterFunds() {
  return (
    <Box p={8} bg="#1A202C" color="white">
      <VStack spacing={8} align="stretch">
        <Heading as="h1" size="2xl" textAlign="center">
          About RichMaster Funds
        </Heading>
        <Text fontSize="xl" textAlign="center">
        RichMaster Funds is a private fund which has had recorded stable return.
        </Text>
      </VStack>
    </Box>
  );
}
