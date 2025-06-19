"use client";

import { useEffect, useState, useCallback } from "react";

import { Client, Thread } from "@langchain/langgraph-sdk";
import { useToast } from "./use-toast";
import { useQueryState } from "nuqs";
import { ENV } from "../config";

// Direct API calls using the correct LangGraph Cloud format

// Direct API calls using the correct format discovered in testing
const apiRequest = async (endpoint: string, options: RequestInit = {}) => {
  const apiUrl = ENV.API_URL;
  const apiKey = ENV.LANGCHAIN_API_KEY;
  
  const response = await fetch(`${apiUrl}${endpoint}`, {
    ...options,
    headers: {
      'X-API-Key': apiKey,
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }
  
  return response.json();
};

export function useThreads(userId: string | undefined) {
  const { toast } = useToast();
  const [isUserThreadsLoading, setIsUserThreadsLoading] = useState(false);
  const [userThreads, setUserThreads] = useState<Thread[]>([]);
  const [threadId, setThreadId] = useQueryState("threadId");

  useEffect(() => {
    if (typeof window == "undefined" || !userId) return;
    getUserThreads(userId);
  }, [userId]); // eslint-disable-line react-hooks/exhaustive-deps

  const createThread = async (id: string) => {
    let thread;
    try {
      console.log(`[threads] createThread for user ${id}`);
      
      // Use direct API call with correct format (POST /threads)
      thread = await apiRequest('/threads', {
        method: 'POST',
        body: JSON.stringify({
          metadata: {
            user_id: id,
          },
        }),
      });
      
      if (!thread || !thread.thread_id) {
        throw new Error("Thread creation failed.");
      }
      setThreadId(thread.thread_id);
      console.log("[threads] created thread", thread.thread_id);
    } catch (e) {
      console.error("Error creating thread", e);
      toast({
        title: "Error creating thread.",
      });
    }
    return thread;
  };

  const getUserThreads = async (id: string) => {
    setIsUserThreadsLoading(true);
    try {
      console.log(`[threads] getUserThreads for user ${id}`);

      // Use direct API call with correct format (POST /threads/search)
      const userThreads = await apiRequest('/threads/search', {
        method: 'POST',
        body: JSON.stringify({
          metadata: {
            user_id: id,
          },
          limit: 100,
          offset: 0,
        }),
      }) as Thread[];

      if (userThreads && userThreads.length > 0) {
        const lastInArray = userThreads[0];
        const allButLast = userThreads.slice(1, userThreads.length);
        const filteredThreads = allButLast.filter(
          (thread) => thread.values && Object.keys(thread.values).length > 0,
        );
        setUserThreads([...filteredThreads, lastInArray]);
      } else {
        setUserThreads([]);
      }
    } catch (error) {
      console.error('[threads] Error fetching user threads:', error);
      
      if (error instanceof Error) {
        if (error.message.includes('403') || error.message.includes('Forbidden')) {
          toast({
            title: "Access Denied", 
            description: "Unable to access threads. Please check your API configuration.",
          });
        } else {
          toast({
            title: "Error fetching threads",
            description: "Unable to load conversation history.",
          });
        }
      }
    } finally {
      setIsUserThreadsLoading(false);
    }
  };

  const getThreadById = async (id: string) => {
    // Use direct API call with correct format (GET /threads/{id})
    return await apiRequest(`/threads/${id}`, {
      method: 'GET',
    }) as Thread;
  };

  const deleteThread = async (id: string, clearMessages: () => void) => {
    if (!userId) {
      throw new Error("User ID not found");
    }
    setUserThreads((prevThreads) => {
      const newThreads = prevThreads.filter(
        (thread) => thread.thread_id !== id,
      );
      return newThreads;
    });
    
    // Use direct API call with correct format (DELETE /threads/{id})
    await apiRequest(`/threads/${id}`, {
      method: 'DELETE',
    });
    
    if (id === threadId) {
      // Remove the threadID from query params, and refetch threads to
      // update the sidebar UI.
      clearMessages();
      getUserThreads(userId);
      setThreadId(null);
    }
  };

  return {
    isUserThreadsLoading,
    userThreads,
    getThreadById,
    setUserThreads,
    getUserThreads,
    createThread,
    deleteThread,
  };
}
