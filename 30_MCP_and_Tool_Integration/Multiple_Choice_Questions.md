# Multiple Choice Questions: MCP and Tool Integration

Test your understanding of Model Context Protocol and tool integration concepts for AI/ML interviews.

---

**Q1. The Model Context Protocol (MCP) primarily solves the problem of:**

A) Training language models faster
B) Fragmented, non-standard tool integrations — providing a unified protocol for connecting LLMs to external tools and data sources
C) Generating images
D) Replacing all APIs with a single endpoint

---

**Q2. MCP's three-layer architecture consists of:**

A) Frontend, backend, database
B) Host (LLM), Client (middleware), and Server (tool provider)
C) Input, hidden, output layers
D) CPU, GPU, TPU

---

**Q3. A tool contract in MCP specifies:**

A) Only the tool's name
B) The tool's name, description, input schema (parameters), output schema (return format), and error codes
C) Only the price of using the tool
D) The LLM's architecture

---

**Q4. MCP improves security over custom integrations by:**

A) Removing all authentication
B) Providing explicit capability declarations, host-side permission enforcement, and audit logging
C) Making all data public
D) Encrypting only the tool name

---

**Q5. The benefit of MCP's transport-agnostic design is that:**

A) It only works with HTTP
B) Tools can communicate over HTTP, WebSocket, stdio, or gRPC without changing their core logic
C) It requires a specific programming language
D) It only works locally

---

**Q6. MCP's standardization enables tool reusability because:**

A) Each LLM needs its own custom tool implementation
B) A tool implementing MCP once works with any MCP-compatible host (Claude, ChatGPT, etc.)
C) Tools cannot be shared across applications
D) Each tool requires a separate protocol

---

**Q7. Resource management in MCP includes:**

A) Only unlimited data transfer
B) Streaming, pagination, caching, rate limiting, timeouts, and resource quotas
C) Only error handling
D) Only authentication

---

**Q8. MCP prompts and templates help LLMs by:**

A) Replacing the LLM's training data
B) Providing guidance on how to correctly use tools, including step-by-step instructions and example calls
C) Removing the need for tools
D) Only handling errors

---

**Q9. Compared to custom API integrations, MCP reduces maintenance burden because:**

A) It requires more code per integration
B) Tool logic lives in one place (the MCP server) rather than being duplicated across each LLM application
C) It has no documentation
D) It cannot be updated

---

**Q10. MCP's tool discovery mechanism allows hosts to:**

A) Only use pre-configured tools
B) Dynamically learn what tools are available, their capabilities, and how to use them at runtime
C) Only use tools at compile time
D) Ignore available tools

---

**Q11. The hub-and-spoke topology in MCP means:**

A) Each tool connects directly to every other tool
B) A single client can connect to multiple tool servers, routing requests to the appropriate server
C) Only one tool can be used at a time
D) Tools must run on the same machine

---

**Q12. MCP supports versioning of tool contracts to:**

A) Break backward compatibility with each update
B) Allow tools to evolve while maintaining backward compatibility with existing hosts
C) Remove old features immediately
D) Force all hosts to update simultaneously

---

**Q13. The host-side permission enforcement in MCP ensures that:**

A) Tools can access anything without restriction
B) Even if a tool declares broad capabilities, the host can restrict which operations are actually permitted for a given agent or user
C) All permissions are granted by default
D) Only the tool decides what is allowed

---

**Q14. Streaming in MCP is important for:**

A) Small, instant responses only
B) Large results (file contents, query results) that are sent incrementally to reduce latency and memory usage
C) Only text generation
D) Only error messages

---

**Q15. The key advantage of MCP over building direct API wrappers for each tool is:**

A) MCP is always faster
B) Ecosystem scalability — as more tools adopt MCP, any MCP-compatible application gains access to all of them without additional integration work
C) MCP requires no server infrastructure
D) Direct API wrappers are always more secure

---

## Answer Key

**Q1. Answer: B**
Without MCP, each LLM application requires custom code for each tool. MCP standardizes the interface so tools are implemented once and work across all compatible applications.

**Q2. Answer: B**
The Host (LLM) makes requests, the Client translates them into MCP protocol messages and routes them, and the Server (tool) executes operations and returns results.

**Q3. Answer: B**
Tool contracts are complete interface specifications including input/output schemas (typically JSON Schema), descriptions, and error codes, enabling validation, discovery, and documentation.

**Q4. Answer: B**
MCP requires tools to explicitly declare capabilities, hosts enforce permission policies, and all invocations are logged. This is far more rigorous than ad-hoc security in custom integrations.

**Q5. Answer: B**
MCP separates the protocol from the transport layer. The same tool can serve over HTTP for cloud, stdio for local development, and WebSocket for streaming, with identical core logic.

**Q6. Answer: B**
A database tool implementing MCP works with Claude, ChatGPT, and any future MCP-compatible host. Tool developers invest once; the tool becomes accessible across the entire ecosystem.

**Q7. Answer: B**
MCP includes mechanisms for efficient resource usage: streaming large results, paginating queries, caching responses, rate limiting, enforcing timeouts, and managing per-agent resource budgets.

**Q8. Answer: B**
MCP prompts provide contextual guidance (best practices, usage examples, warnings) that are included in the LLM's context, helping it use tools correctly and avoid common mistakes.

**Q9. Answer: B**
With MCP, tool logic is centralized in one MCP server. Without MCP, the same logic must be reimplemented (with potential inconsistencies) for each LLM application that uses the tool.

**Q10. Answer: B**
MCP's discovery protocol lets hosts query connected servers for available tools, their schemas, and descriptions at runtime, enabling dynamic adaptation without hardcoded tool configurations.

**Q11. Answer: B**
One client connects to multiple servers (database server, file server, Slack server), routing each tool call to the appropriate server. This isolates failures and enables modular tool management.

**Q12. Answer: B**
Contract versioning allows tools to add new capabilities while maintaining support for older hosts, ensuring the ecosystem can evolve gradually without breaking existing integrations.

**Q13. Answer: B**
The host acts as a policy enforcement point. A file system tool may declare read/write capabilities, but the host can enforce read-only access for a particular agent or user role.

**Q14. Answer: B**
For large datasets or file contents, streaming sends data incrementally rather than buffering the entire result. This reduces time-to-first-byte and prevents memory exhaustion.

**Q15. Answer: B**
MCP creates network effects: each new MCP tool benefits all MCP hosts, and each new MCP host benefits from all existing tools. Direct API wrappers create O(n×m) integration work instead of O(n+m).

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*
