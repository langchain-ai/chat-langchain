/**
 * Self-hosted Express server for the retrieval graph.
 *
 * This server provides REST API endpoints for invoking the graph,
 * with support for streaming, checkpointing, and thread management.
 */

import express, { Request, Response } from "express";
import cors from "cors";
import { config as loadEnv } from "dotenv";
import { HumanMessage } from "@langchain/core/messages";
import { graph } from "./retrieval_graph/graph.js";

// Load environment variables
loadEnv();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Request logging middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} ${req.method} ${req.path}`);
  next();
});

/**
 * Health check endpoint
 */
app.get("/health", (req: Request, res: Response) => {
  res.json({ status: "healthy", timestamp: new Date().toISOString() });
});

/**
 * POST /runs
 * Invoke the graph with a new question
 *
 * Body:
 * {
 *   "messages": ["What is LangChain?"],
 *   "thread_id": "optional-thread-id",
 *   "config": {} // optional configuration
 * }
 */
app.post("/runs", async (req: Request, res: Response) => {
  try {
    const { messages, thread_id, config = {} } = req.body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: "messages array is required" });
    }

    // Convert string messages to HumanMessage objects
    const messageObjects = messages.map((msg: string | any) => {
      if (typeof msg === "string") {
        return new HumanMessage(msg);
      }
      return msg;
    });

    // Prepare config
    const runnableConfig = {
      ...config,
      configurable: {
        ...config.configurable,
        thread_id: thread_id || `thread_${Date.now()}`,
      },
    };

    // Invoke the graph
    const result = await graph.invoke(
      { messages: messageObjects },
      runnableConfig
    );

    res.json({
      success: true,
      thread_id: runnableConfig.configurable.thread_id,
      result,
    });
  } catch (error: any) {
    console.error("Error in /runs:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error.message,
    });
  }
});

/**
 * POST /runs/stream
 * Stream the graph execution (Server-Sent Events)
 *
 * Body:
 * {
 *   "messages": ["What is LangChain?"],
 *   "thread_id": "optional-thread-id",
 *   "config": {} // optional configuration
 * }
 */
app.post("/runs/stream", async (req: Request, res: Response) => {
  try {
    const { messages, thread_id, config = {} } = req.body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: "messages array is required" });
    }

    // Set headers for Server-Sent Events
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    // Convert string messages to HumanMessage objects
    const messageObjects = messages.map((msg: string | any) => {
      if (typeof msg === "string") {
        return new HumanMessage(msg);
      }
      return msg;
    });

    // Prepare config
    const runnableConfig = {
      ...config,
      configurable: {
        ...config.configurable,
        thread_id: thread_id || `thread_${Date.now()}`,
      },
    };

    // Stream the graph execution
    const stream = await graph.stream(
      { messages: messageObjects },
      runnableConfig
    );

    // Send events to client
    for await (const event of stream) {
      res.write(`data: ${JSON.stringify(event)}\n\n`);
    }

    // End the stream
    res.write("data: [DONE]\n\n");
    res.end();
  } catch (error: any) {
    console.error("Error in /runs/stream:", error);
    if (!res.headersSent) {
      res.status(500).json({
        error: "Internal server error",
        message: error.message,
      });
    } else {
      res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
      res.end();
    }
  }
});

/**
 * GET /runs/:run_id
 * Get the status of a specific run
 */
app.get("/runs/:run_id", async (req: Request, res: Response) => {
  try {
    const { run_id } = req.params;

    // Note: This would require storing run information
    // For now, return a placeholder
    res.json({
      run_id,
      status: "completed",
      message: "Run status tracking not yet implemented in self-hosted mode",
    });
  } catch (error: any) {
    console.error("Error in /runs/:run_id:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error.message,
    });
  }
});

/**
 * POST /threads/:thread_id/runs
 * Continue a conversation in an existing thread
 *
 * Body:
 * {
 *   "messages": ["Follow-up question"],
 *   "config": {} // optional configuration
 * }
 */
app.post("/threads/:thread_id/runs", async (req: Request, res: Response) => {
  try {
    const { thread_id } = req.params;
    const { messages, config = {} } = req.body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: "messages array is required" });
    }

    // Convert string messages to HumanMessage objects
    const messageObjects = messages.map((msg: string | any) => {
      if (typeof msg === "string") {
        return new HumanMessage(msg);
      }
      return msg;
    });

    // Prepare config with thread_id
    const runnableConfig = {
      ...config,
      configurable: {
        ...config.configurable,
        thread_id,
      },
    };

    // Invoke the graph
    const result = await graph.invoke(
      { messages: messageObjects },
      runnableConfig
    );

    res.json({
      success: true,
      thread_id,
      result,
    });
  } catch (error: any) {
    console.error("Error in /threads/:thread_id/runs:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error.message,
    });
  }
});

/**
 * GET /threads/:thread_id/state
 * Get the current state of a thread
 */
app.get("/threads/:thread_id/state", async (req: Request, res: Response) => {
  try {
    const { thread_id } = req.params;

    // Note: This would require implementing checkpointing
    // For now, return a placeholder
    res.json({
      thread_id,
      state: {},
      message: "Thread state retrieval not yet implemented in self-hosted mode",
      note: "Implement checkpointing using @langchain/langgraph-checkpoint-postgres or similar",
    });
  } catch (error: any) {
    console.error("Error in /threads/:thread_id/state:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error.message,
    });
  }
});

/**
 * Error handling middleware
 */
app.use((err: any, req: Request, res: Response, next: any) => {
  console.error("Unhandled error:", err);
  res.status(500).json({
    error: "Internal server error",
    message: err.message,
  });
});

/**
 * Start the server
 */
app.listen(PORT, () => {
  console.log(`
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🚀 Chat LangChain Backend Server (Self-Hosted)        ║
║                                                           ║
║   Status: Running                                         ║
║   Port:   ${PORT}                                         ║
║   URL:    http://localhost:${PORT}                       ║
║                                                           ║
║   Endpoints:                                              ║
║   - GET  /health                                          ║
║   - POST /runs                                            ║
║   - POST /runs/stream                                     ║
║   - GET  /runs/:run_id                                    ║
║   - POST /threads/:thread_id/runs                         ║
║   - GET  /threads/:thread_id/state                        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
  `);

  console.log("\n📝 Note: For production deployment, consider:");
  console.log("   - Adding authentication/authorization");
  console.log("   - Implementing rate limiting");
  console.log("   - Setting up checkpointing with PostgreSQL");
  console.log("   - Adding request validation middleware");
  console.log("   - Configuring logging and monitoring");
  console.log("\n💡 Tip: Use LangGraph Cloud for managed deployment with these features built-in.\n");
});

// Graceful shutdown
process.on("SIGTERM", () => {
  console.log("\nReceived SIGTERM, shutting down gracefully...");
  process.exit(0);
});

process.on("SIGINT", () => {
  console.log("\nReceived SIGINT, shutting down gracefully...");
  process.exit(0);
});

