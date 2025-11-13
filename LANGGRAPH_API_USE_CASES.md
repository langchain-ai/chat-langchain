# LangGraph API: Purpose and Use Cases

This document explains the purpose, use cases, and when to use each endpoint category in the LangGraph API.

---

## 1. Assistants (Create, Search, Update, Delete)

### Purpose
**Assistants** are reusable configurations of LangGraph workflows. They define how a graph should behave, what models to use, what parameters to apply, and any static context. Think of an assistant as a "template" or "blueprint" for running a specific type of AI workflow.

### Key Concepts
- **Graph ID**: References a graph defined in your `langgraph.json`
- **Config**: Runtime configuration (model selection, temperature, etc.)
- **Context**: Static data that's always available to the assistant
- **Versions**: Assistants can have multiple versions for A/B testing or gradual rollouts

### Use Cases

#### 1.1 Create Assistant
**When to use:**
- Setting up a new AI workflow for your application
- Creating different assistant variants (e.g., "customer-support", "technical-assistant")
- Initializing assistants with specific configurations (different models, parameters)
- Creating assistants with static context (knowledge base, instructions, system prompts)

**Example scenarios:**
```python
# Create a customer support assistant
POST /assistants
{
  "graph_id": "chat",
  "name": "Customer Support Bot",
  "config": {
    "configurable": {
      "model": "gpt-4",
      "temperature": 0.7
    }
  },
  "context": {
    "company_name": "Acme Corp",
    "support_policies": "Always be polite and helpful"
  },
  "metadata": {
    "department": "support",
    "language": "en"
  }
}

# Create a technical documentation assistant
POST /assistants
{
  "graph_id": "chat",
  "name": "Tech Docs Assistant",
  "config": {
    "configurable": {
      "model": "gpt-4-turbo",
      "temperature": 0.3  # Lower temperature for more factual responses
    }
  },
  "context": {
    "documentation_urls": ["https://docs.example.com"],
    "code_examples": true
  }
}
```

#### 1.2 Search Assistants
**When to use:**
- Listing all assistants in your system
- Finding assistants by metadata (e.g., "all support assistants")
- Filtering assistants by graph type
- Building admin dashboards or management UIs

**Example scenarios:**
```python
# Find all customer support assistants
POST /assistants/search
{
  "metadata": {"department": "support"},
  "limit": 50
}

# List all assistants using a specific graph
POST /assistants/search
{
  "graph_id": "chat",
  "sort_by": "created_at",
  "sort_order": "desc"
}
```

#### 1.3 Update Assistant
**When to use:**
- Updating assistant configuration without creating a new version
- Changing static context (e.g., updating knowledge base)
- Modifying metadata tags
- Adjusting model parameters or settings

**Example scenarios:**
```python
# Update assistant's context with new information
PATCH /assistants/{assistant_id}
{
  "context": {
    "updated_knowledge_base": "2024-01-15",
    "new_features": ["feature1", "feature2"]
  }
}

# Change assistant name and description
PATCH /assistants/{assistant_id}
{
  "name": "Updated Assistant Name",
  "description": "New description"
}
```

#### 1.4 Delete Assistant
**When to use:**
- Removing deprecated assistants
- Cleaning up test assistants
- Removing assistants that are no longer needed
- **Note**: Deletes all versions of the assistant

**Example scenarios:**
```python
# Delete a test assistant
DELETE /assistants/{test_assistant_id}

# Clean up old assistants (after archiving data)
DELETE /assistants/{old_assistant_id}
```

### Best Practices
- Create assistants once and reuse them across multiple threads
- Use metadata to organize and filter assistants
- Use versions for gradual rollouts (create new version, test, then set as latest)
- Store static, reusable data in `context` rather than passing it with each run

---

## 2. Threads (State Management, History, Streaming)

### Purpose
**Threads** are conversation sessions that maintain state and context across multiple interactions. They're the "memory" of your application, preserving conversation history, user preferences, and any accumulated state from previous runs.

### Key Concepts
- **State Persistence**: Threads store state that persists between runs
- **Checkpoints**: Snapshots of state at specific points in time
- **History**: Complete record of all state changes
- **Streaming**: Real-time updates as the thread processes runs

### Use Cases

#### 2.1 Create Thread
**When to use:**
- Starting a new conversation session
- Initializing a user's interaction context
- Setting up a thread with initial state or metadata
- Creating threads with TTL for automatic cleanup

**Example scenarios:**
```python
# Start a new customer conversation
POST /threads
{
  "metadata": {
    "user_id": "user-123",
    "session_id": "session-456",
    "channel": "web"
  },
  "ttl": {
    "strategy": "delete",
    "ttl": 10080  # Delete after 7 days of inactivity
  }
}

# Create thread with initial state
POST /threads
{
  "metadata": {"conversation_type": "support"},
  "supersteps": [{
    "as_node": "initialize",
    "values": {
      "user_preferences": {"language": "en"},
      "conversation_history": []
    }
  }]
}
```

#### 2.2 Get Thread State
**When to use:**
- Retrieving current conversation state
- Checking what information the thread remembers
- Inspecting thread status (idle, busy, interrupted)
- Getting the latest checkpoint for resumption

**Example scenarios:**
```python
# Check thread status before sending new message
GET /threads/{thread_id}/state

# Get state including subgraphs
GET /threads/{thread_id}/state?subgraphs=true
```

#### 2.3 Update Thread State
**When to use:**
- Manually updating conversation state
- Injecting external data into the thread
- Correcting or modifying thread state
- Resetting or clearing certain state values

**Example scenarios:**
```python
# Update user preferences mid-conversation
POST /threads/{thread_id}/state
{
  "values": {
    "user_preferences": {
      "language": "es",  # User switched to Spanish
      "theme": "dark"
    }
  }
}

# Inject external data (e.g., from database)
POST /threads/{thread_id}/state
{
  "values": {
    "user_profile": {
      "name": "John Doe",
      "subscription": "premium"
    }
  },
  "as_node": "load_user_data"
}
```

#### 2.4 Get Thread History
**When to use:**
- Reviewing conversation history
- Debugging what happened in previous runs
- Building conversation replay features
- Analyzing conversation patterns
- Implementing "undo" functionality

**Example scenarios:**
```python
# Get last 20 checkpoints
GET /threads/{thread_id}/history?limit=20

# Get history before a specific checkpoint
POST /threads/{thread_id}/history
{
  "limit": 10,
  "before": {
    "checkpoint_id": "checkpoint-123"
  }
}

# Filter history by metadata
POST /threads/{thread_id}/history
{
  "limit": 50,
  "metadata": {"run_type": "user_message"}
}
```

#### 2.5 Join Thread Stream
**When to use:**
- Real-time chat applications
- Streaming responses to users
- Monitoring thread activity
- Building live dashboards
- Implementing typing indicators

**Example scenarios:**
```python
# Stream all events from a thread
GET /threads/{thread_id}/stream

# Stream only lifecycle events (start/end)
GET /threads/{thread_id}/stream?stream_modes=lifecycle

# Resume streaming from last event
GET /threads/{thread_id}/stream
Headers: Last-Event-ID: event-123
```

#### 2.6 Copy Thread
**When to use:**
- Creating conversation branches
- Testing different approaches on same context
- Forking conversations for different users
- Creating backups before major changes

**Example scenarios:**
```python
# Branch conversation for testing
POST /threads/{thread_id}/copy
# Returns new thread with same state

# Create backup before risky operation
POST /threads/{thread_id}/copy
# Use backup thread if something goes wrong
```

### Best Practices
- Create one thread per user session/conversation
- Use TTLs to automatically clean up old threads
- Store user identifiers in metadata for easy lookup
- Use checkpoints to enable conversation resumption
- Stream for real-time user experiences

---

## 3. Thread Runs (Execution on Threads)

### Purpose
**Thread Runs** execute a graph/assistant on an existing thread, updating the thread's state. They're the mechanism for processing user inputs and generating responses while maintaining conversation context.

### Key Concepts
- **Stateful Execution**: Runs update the thread's persistent state
- **Streaming**: Can stream intermediate results
- **Interrupts**: Can pause execution at specific nodes
- **Multitask Strategy**: How to handle concurrent runs

### Use Cases

#### 3.1 Create Background Run
**When to use:**
- Fire-and-forget operations
- Long-running tasks where you don't need immediate results
- Batch processing
- When you'll poll for results later

**Example scenarios:**
```python
# Send user message, don't wait for response
POST /threads/{thread_id}/runs
{
  "assistant_id": "assistant-123",
  "input": {
    "messages": [{"role": "user", "content": "Hello!"}]
  }
}
# Returns immediately with run_id
# Poll GET /threads/{thread_id}/runs/{run_id} for status
```

#### 3.2 Create Run, Stream Output
**When to use:**
- Real-time chat applications
- Showing progressive results to users
- Implementing typing indicators
- Streaming long-form responses

**Example scenarios:**
```python
# Stream response as it's generated
POST /threads/{thread_id}/runs/stream
{
  "assistant_id": "assistant-123",
  "input": {"messages": [{"role": "user", "content": "Tell me a story"}]},
  "stream_mode": ["values", "messages"]
}
# Returns SSE stream with incremental updates
```

#### 3.3 Create Run, Wait for Output
**When to use:**
- Synchronous operations
- When you need the final result before proceeding
- Simple request-response patterns
- Testing and debugging

**Example scenarios:**
```python
# Wait for complete response
POST /threads/{thread_id}/runs/wait
{
  "assistant_id": "assistant-123",
  "input": {"messages": [{"role": "user", "content": "What's 2+2?"}]}
}
# Returns only after run completes
```

#### 3.4 Join Run Stream
**When to use:**
- Reconnecting to an existing run's stream
- Resuming after network interruption
- Monitoring a run that's already in progress
- Building resumable streaming clients

**Example scenarios:**
```python
# Resume streaming from last event
GET /threads/{thread_id}/runs/{run_id}/stream
Headers: Last-Event-ID: event-456

# Monitor run progress
GET /threads/{thread_id}/runs/{run_id}/stream?stream_mode=lifecycle
```

#### 3.5 Cancel Run
**When to use:**
- User cancels request
- Timeout handling
- Stopping erroneous runs
- Cleanup operations

**Example scenarios:**
```python
# Cancel and interrupt
POST /threads/{thread_id}/runs/{run_id}/cancel?action=interrupt

# Cancel and rollback (delete run and checkpoints)
POST /threads/{thread_id}/runs/{run_id}/cancel?action=rollback&wait=true
```

### Best Practices
- Use streaming for better UX in chat applications
- Use background runs for long operations
- Set appropriate multitask strategies (enqueue vs reject)
- Handle interrupts for user interaction points
- Monitor run status for error handling

---

## 4. Stateless Runs (Temporary Execution)

### Purpose
**Stateless Runs** execute a graph/assistant without maintaining persistent state. They create a temporary thread, execute, and optionally delete it afterward. Perfect for one-off operations that don't need conversation context.

### Key Concepts
- **No Persistent State**: Each run is independent
- **Temporary Threads**: Created automatically, can be auto-deleted
- **One-off Operations**: No conversation history needed

### Use Cases

#### 4.1 Create Stateless Run
**When to use:**
- One-time queries that don't need context
- Batch processing of independent tasks
- Testing graph behavior without state
- Simple request-response without conversation history

**Example scenarios:**
```python
# One-time translation request
POST /runs
{
  "assistant_id": "translator-assistant",
  "input": {
    "text": "Hello world",
    "target_language": "es"
  },
  "on_completion": "delete"  # Clean up thread after
}

# Batch process multiple independent tasks
POST /runs/batch
[
  {"assistant_id": "analyzer", "input": {"data": "task1"}},
  {"assistant_id": "analyzer", "input": {"data": "task2"}},
  {"assistant_id": "analyzer", "input": {"data": "task3"}}
]
```

#### 4.2 Stateless Run with Streaming
**When to use:**
- Streaming one-off operations
- Real-time processing without state
- Progressive result display

**Example scenarios:**
```python
# Stream analysis results
POST /runs/stream
{
  "assistant_id": "analyzer-assistant",
  "input": {"document": "..."},
  "stream_mode": ["values"],
  "on_completion": "delete"
}
```

### When to Use Stateless vs Thread Runs

**Use Stateless Runs when:**
- ✅ Each request is independent
- ✅ No conversation history needed
- ✅ One-off operations
- ✅ Batch processing
- ✅ Testing/debugging

**Use Thread Runs when:**
- ✅ Conversation context matters
- ✅ User sessions need persistence
- ✅ Multi-turn interactions
- ✅ State accumulation needed
- ✅ Resumable conversations

### Best Practices
- Use `on_completion: "delete"` to clean up temporary threads
- Use for batch operations where each item is independent
- Don't use for conversations that need context

---

## 5. Crons (Scheduled Runs)

### Purpose
**Crons** schedule periodic runs that execute on a schedule. They can run on new threads (isolated) or share state in an existing thread. Perfect for automated tasks, periodic reports, and scheduled workflows.

### Key Concepts
- **Schedule**: Cron format (e.g., "0 0 * * *" for daily at midnight)
- **End Time**: When to stop scheduling
- **Isolated vs Shared**: Can run on new threads or same thread

### Use Cases

#### 5.1 Create Cron (Isolated Runs)
**When to use:**
- Daily/weekly reports
- Periodic data processing
- Scheduled notifications
- Automated tasks that don't need shared state

**Example scenarios:**
```python
# Daily report generation
POST /runs/crons
{
  "assistant_id": "report-generator",
  "schedule": "0 9 * * *",  # 9 AM daily
  "end_time": "2024-12-31T23:59:59Z",
  "input": {
    "report_type": "daily_summary"
  },
  "metadata": {"type": "scheduled_report"}
}
```

#### 5.2 Create Thread Cron (Shared State)
**When to use:**
- Periodic updates to same conversation
- Maintaining state across scheduled runs
- Continuous monitoring with history
- Workflows that build on previous runs

**Example scenarios:**
```python
# Periodic status updates on same thread
POST /threads/{thread_id}/runs/crons
{
  "assistant_id": "monitor-assistant",
  "schedule": "*/5 * * * *",  # Every 5 minutes
  "end_time": "2024-12-31T23:59:59Z",
  "input": {
    "check_type": "system_health"
  }
}
# All runs share the same thread, building history
```

### Use Cases

1. **Daily Reports**: Generate and send daily summaries
2. **Health Checks**: Periodic system monitoring
3. **Data Sync**: Regular data synchronization tasks
4. **Reminders**: Scheduled notifications
5. **Cleanup Tasks**: Periodic cleanup operations
6. **Analytics**: Scheduled data analysis

### Best Practices
- Set appropriate `end_time` to prevent infinite scheduling
- Use isolated crons for independent tasks
- Use thread crons when state needs to accumulate
- Monitor cron execution with search endpoints
- Clean up completed crons when no longer needed

---

## 6. Store (Key-Value Storage)

### Purpose
The **Store** provides persistent, cross-thread key-value storage. It's like a global database that any thread can access, perfect for storing user preferences, shared knowledge, or any data that needs to persist across conversations.

### Key Concepts
- **Namespaces**: Hierarchical organization (like directories)
- **Cross-Thread**: Accessible from any thread
- **Persistent**: Survives thread deletion
- **Searchable**: Can search by namespace, filters, or semantic search

### Use Cases

#### 6.1 Store User Preferences
**When to use:**
- Storing user settings that persist across conversations
- User profiles and preferences
- Cross-session data

**Example scenarios:**
```python
# Store user preferences
PUT /store/items
{
  "namespace": ["users", "preferences"],
  "key": "user-123",
  "value": {
    "language": "en",
    "theme": "dark",
    "notifications": true
  }
}

# Retrieve user preferences
GET /store/items?key=user-123&namespace[]=users&namespace[]=preferences
```

#### 6.2 Store Shared Knowledge
**When to use:**
- Company-wide knowledge base
- Shared documentation
- Cross-conversation facts

**Example scenarios:**
```python
# Store company policies
PUT /store/items
{
  "namespace": ["knowledge", "policies"],
  "key": "refund_policy",
  "value": {
    "title": "Refund Policy",
    "content": "30-day money-back guarantee...",
    "last_updated": "2024-01-15"
  }
}

# Search for policies
POST /store/items/search
{
  "namespace_prefix": ["knowledge", "policies"],
  "query": "refund",
  "limit": 10
}
```

#### 6.3 Store Conversation Summaries
**When to use:**
- Storing conversation summaries for quick access
- Building user history
- Cross-thread context

**Example scenarios:**
```python
# Store conversation summary
PUT /store/items
{
  "namespace": ["users", "conversations"],
  "key": "user-123-summary-2024-01",
  "value": {
    "summary": "User asked about product features...",
    "topics": ["pricing", "features"],
    "sentiment": "positive"
  }
}
```

#### 6.4 Store Feature Flags / Configuration
**When to use:**
- Application configuration
- Feature flags
- A/B test configurations

**Example scenarios:**
```python
# Store feature flags
PUT /store/items
{
  "namespace": ["config", "features"],
  "key": "new_ui_enabled",
  "value": {
    "enabled": true,
    "rollout_percentage": 50,
    "user_ids": ["user-1", "user-2"]
  }
}
```

### Use Cases Summary

1. **User Data**: Preferences, profiles, settings
2. **Knowledge Base**: Shared information, documentation
3. **Conversation History**: Summaries, important facts
4. **Configuration**: Feature flags, app settings
5. **Cross-Thread Context**: Data shared across conversations
6. **Caching**: Store computed results for reuse

### Best Practices
- Use hierarchical namespaces for organization
- Use semantic search for finding related items
- Set appropriate TTLs if needed (via thread TTL)
- Use filters for efficient queries
- Store structured data (JSON objects) as values

---

## 7. A2A (Agent-to-Agent Protocol)

### Purpose
**A2A (Agent-to-Agent Protocol)** exposes assistants as standardized agents that can communicate with other A2A-compliant systems. It uses JSON-RPC 2.0 for standardized agent communication.

### Key Concepts
- **JSON-RPC 2.0**: Standardized protocol
- **Interoperability**: Works with other A2A systems
- **Threaded Conversations**: Supports conversation context
- **Task-Based**: Asynchronous task execution

### Use Cases

#### 7.1 Agent-to-Agent Communication
**When to use:**
- Integrating with other AI agent systems
- Multi-agent workflows
- Agent orchestration
- Standardized agent interfaces

**Example scenarios:**
```python
# Send message to another agent
POST /a2a/{assistant_id}
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{
        "kind": "text",
        "text": "Analyze this data"
      }],
      "messageId": "msg-1"
    },
    "thread": {
      "threadId": "thread-123"
    }
  }
}

# Check task status
POST /a2a/{assistant_id}
{
  "jsonrpc": "2.0",
  "id": "req-2",
  "method": "tasks/get",
  "params": {
    "taskId": "task-456"
  }
}
```

### Use Cases

1. **Multi-Agent Systems**: Agents communicating with each other
2. **Agent Marketplaces**: Exposing agents via standard protocol
3. **Agent Orchestration**: Coordinating multiple agents
4. **Integration**: Connecting with A2A-compliant tools
5. **Workflow Automation**: Agent-based workflows

### Best Practices
- Use for standardized agent interfaces
- Implement proper error handling for JSON-RPC
- Use thread context for multi-turn agent conversations
- Handle task status polling appropriately

---

## 8. MCP (Model Context Protocol)

### Purpose
**MCP (Model Context Protocol)** exposes assistants as MCP servers, enabling integration with MCP-compatible clients and tools. It's a protocol for standardized AI model interactions.

### Key Concepts
- **Stateless**: Sessions are not persisted
- **JSON-RPC 2.0**: Uses JSON-RPC for communication
- **Streaming Support**: Can stream responses
- **Tool Integration**: Works with MCP tools

### Use Cases

#### 8.1 MCP Client Integration
**When to use:**
- Integrating with MCP-compatible clients
- Exposing assistants via MCP protocol
- Tool integration through MCP
- Standardized model access

**Example scenarios:**
```python
# Initialize MCP session
POST /mcp/
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "initialize",
  "params": {
    "clientInfo": {
      "name": "my-client",
      "version": "1.0.0"
    },
    "protocolVersion": "2024-11-05",
    "capabilities": {}
  }
}

# Send request to MCP server
POST /mcp/
{
  "jsonrpc": "2.0",
  "id": "2",
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {"query": "python"}
  }
}
```

### Use Cases

1. **MCP Client Integration**: Connect MCP clients to your assistants
2. **Tool Integration**: Use MCP-compatible tools
3. **Standardized Access**: Expose assistants via MCP protocol
4. **Development Tools**: Integrate with MCP development tools

### Best Practices
- Use for MCP-specific integrations
- Handle stateless nature (no session persistence)
- Implement proper MCP protocol compliance
- Use streaming for better performance

---

## 9. System (Health Checks, Metrics)

### Purpose
**System endpoints** provide operational monitoring, health checks, and server information. Essential for production deployments and monitoring.

### Key Concepts
- **Health Checks**: Verify system is running
- **Metrics**: Performance and operational metrics
- **Server Info**: Version and feature information

### Use Cases

#### 9.1 Health Check
**When to use:**
- Load balancer health checks
- Monitoring system availability
- Automated alerting
- Deployment verification

**Example scenarios:**
```python
# Basic health check
GET /ok
# Returns: {"ok": true}

# Health check with database connectivity
GET /ok?check_db=1
# Verifies database connection too
```

#### 9.2 Server Information
**When to use:**
- Version checking
- Feature detection
- Debugging
- API discovery

**Example scenarios:**
```python
# Get server info
GET /info
# Returns: {
#   "version": "0.1.0",
#   "langgraph_py_version": "x.x.x",
#   "flags": {...},
#   "metadata": {...}
# }
```

#### 9.3 System Metrics
**When to use:**
- Performance monitoring
- Prometheus integration
- Operational dashboards
- Capacity planning

**Example scenarios:**
```python
# Get Prometheus metrics
GET /metrics?format=prometheus
# Returns Prometheus-formatted metrics

# Get JSON metrics
GET /metrics?format=json
# Returns structured JSON metrics
```

### Use Cases

1. **Health Monitoring**: Regular health checks for uptime monitoring
2. **Load Balancing**: Health endpoints for load balancers
3. **Metrics Collection**: Prometheus/Grafana integration
4. **Debugging**: Version and feature information
5. **Alerting**: Automated alerts based on health status

### Best Practices
- Set up regular health check monitoring
- Integrate metrics with monitoring systems (Prometheus, Datadog, etc.)
- Use database health checks for critical deployments
- Monitor metrics for capacity planning

---

## Summary: When to Use Each Endpoint Category

| Endpoint Category | Use When | Don't Use When |
|------------------|----------|----------------|
| **Assistants** | Setting up reusable workflows, configuring graphs | One-off configurations |
| **Threads** | Conversations, stateful interactions, multi-turn | One-off requests, stateless operations |
| **Thread Runs** | Executing on existing conversations | Independent operations |
| **Stateless Runs** | One-off operations, batch processing | Conversations needing context |
| **Crons** | Scheduled tasks, periodic operations | Real-time user interactions |
| **Store** | Cross-thread data, user preferences, shared knowledge | Thread-specific state (use thread state) |
| **A2A** | Agent-to-agent communication, multi-agent systems | Direct user interactions |
| **MCP** | MCP client integration, tool integration | General API usage |
| **System** | Monitoring, health checks, operations | Application logic |

---

## Common Patterns

### Pattern 1: Chat Application
```
1. Create Assistant (once)
2. Create Thread per user session
3. Use Thread Runs with streaming for messages
4. Store user preferences in Store
5. Monitor with System endpoints
```

### Pattern 2: Batch Processing
```
1. Create Assistant (once)
2. Use Stateless Runs for each item
3. Use Batch endpoint for parallel processing
4. Monitor with System metrics
```

### Pattern 3: Scheduled Reports
```
1. Create Assistant (once)
2. Create Cron for scheduled execution
3. Use Store for report templates/config
4. Monitor Cron execution
```

### Pattern 4: Multi-Agent System
```
1. Create multiple Assistants (one per agent)
2. Use A2A protocol for agent communication
3. Use Threads for agent conversations
4. Use Store for shared agent knowledge
```

---

*This document provides guidance on when and why to use each endpoint category. For detailed API specifications, refer to the main API documentation.*

