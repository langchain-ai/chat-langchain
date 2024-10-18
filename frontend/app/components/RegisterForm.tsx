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

export function RegisterForm() {
  return (
    <Box p={8} bg="#1A202C" color="white" height="100%">
      <VStack spacing={6} align="center" justify="center" height="100%">
        <Heading as="h1" size="2xl" textAlign="center" mb={8}>
          Register
        </Heading>
        <Box width="100%" maxWidth="400px">
          <FormControl id="email" mb={4}>
            <FormLabel>Email address</FormLabel>
            <Input type="email" placeholder="Enter your email" />
          </FormControl>
          <FormControl id="password" mb={4}>
            <FormLabel>Password</FormLabel>
            <Input type="password" placeholder="Enter your password" />
          </FormControl>
          <FormControl id="confirmPassword" mb={8}>
            <FormLabel>Confirm Password</FormLabel>
            <Input type="password" placeholder="Confirm your password" />
          </FormControl>
          <Button
            colorScheme="blue"
            size="lg"
            width="100%"
            mb={4}
            bg="#3182CE"
            _hover={{ bg: "#2B6CB0" }}
          >
            Register
          </Button>
          <Text fontSize="sm" textAlign="center">
            Already have an account?{" "}
            <Text as="a" color="blue.400" href="/login">
              Login
            </Text>
          </Text>
        </Box>
      </VStack>
    </Box>
  );
}
