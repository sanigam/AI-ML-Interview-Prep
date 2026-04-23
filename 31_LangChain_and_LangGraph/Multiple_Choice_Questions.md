# Multiple Choice Questions: LangChain and LangGraph

Test your understanding of LangChain and LangGraph concepts for AI/ML interviews.

---

**Q1. LangChain primarily solves the problem of:**

A) Training LLMs from scratch
B) Providing reusable abstractions for building LLM applications (prompts, chains, memory, document processing)
C) Replacing all APIs
D) Only generating images

---

**Q2. In LangChain, a "chain" differs from an "agent" because:**

A) Chains use reinforcement learning
B) Chains follow a deterministic, predefined sequence of steps, while agents let the LLM decide what to do next iteratively
C) Agents cannot use tools
D) Chains are always faster than agents

---

**Q3. LCEL (LangChain Expression Language) composes chains using:**

A) SQL queries
B) Pipe operators (|) for declarative, functional composition of LLM calls and tools
C) Only Python classes
D) XML configuration files

---

**Q4. LangGraph extends LangChain by providing:**

A) Only simple linear chains
B) Graph-based stateful orchestration for complex multi-step agent workflows with persistence and human-in-the-loop
C) A replacement for Python
D) Only document retrieval

---

**Q5. In LangGraph, nodes represent:**

A) Database tables
B) Individual processing steps (LLM calls, tool invocations, logic) connected by conditional edges
C) Only input data
D) Hardware components

---

**Q6. Memory management in LangChain helps agents by:**

A) Increasing the model size
B) Maintaining conversation history and context across interactions, enabling multi-turn conversations
C) Removing all previous context
D) Only storing user credentials

---

**Q7. LangChain's document loaders and text splitters are used for:**

A) Training new models
B) Loading documents from various sources and splitting them into chunks for RAG pipelines
C) Only PDF generation
D) Image processing

---

**Q8. Human-in-the-loop in LangGraph allows:**

A) Removing humans from the process entirely
B) Pausing agent workflows for human review or approval before proceeding with critical actions
C) Only manual data entry
D) Retraining the model during execution

---

**Q9. State persistence in LangGraph enables:**

A) Only in-memory computation
B) Saving and resuming long-running workflows across sessions, supporting checkpoints and recovery
C) Deleting all state after each step
D) Only synchronous execution

---

**Q10. Output parsers in LangChain are used to:**

A) Generate random outputs
B) Convert LLM text outputs into structured formats (JSON, lists, Pydantic objects) for downstream processing
C) Only format text for display
D) Translate between languages

---

**Q11. The advantage of using LangChain's retrieval chain over building RAG from scratch is:**

A) It's always more accurate
B) It provides pre-built components for document loading, splitting, embedding, retrieval, and prompt formatting, reducing boilerplate code
C) It only works with OpenAI models
D) It eliminates the need for a vector database

---

**Q12. Conditional edges in LangGraph enable:**

A) Only linear execution
B) Dynamic routing where the next node depends on the current state or LLM output, enabling branching workflows
C) Random node selection
D) Only backward traversal

---

**Q13. LangChain's callback system provides:**

A) Only error handling
B) Hooks for logging, monitoring, streaming, and debugging at each step of chain/agent execution
C) Only authentication
D) Data encryption

---

**Q14. When should you choose LangGraph over simple LangChain chains?**

A) For all tasks regardless of complexity
B) When workflows require complex state management, conditional branching, loops, or human-in-the-loop approval
C) Only for text classification
D) When no tools are needed

---

**Q15. LangChain's tool abstraction standardizes:**

A) Only database access
B) The interface for LLMs to discover, invoke, and process results from external tools and APIs
C) Only file I/O operations
D) Model training procedures

---

## Answer Key

**Q1. Answer: B**
LangChain abstracts common LLM patterns (prompt templates, chain composition, memory, document processing) into reusable components, letting developers focus on application logic rather than boilerplate.

**Q2. Answer: B**
Chains execute steps in a fixed order (prompt → LLM → parser). Agents use the LLM to decide which tools to call and in what order, adapting their approach based on intermediate results.

**Q3. Answer: B**
LCEL uses pipe operators: `chain = prompt | llm | parser`. This declarative style enables readability, streaming support, parallel execution detection, and functional composition.

**Q4. Answer: B**
LangGraph models workflows as directed graphs with state, enabling complex patterns like loops, conditional branching, persistent state across sessions, and human approval steps.

**Q5. Answer: B**
Each LangGraph node performs a computation (LLM call, tool execution, logic). Edges connect nodes and can be conditional, enabling dynamic routing based on state.

**Q6. Answer: B**
LangChain memory modules (ConversationBufferMemory, ConversationSummaryMemory) store and retrieve conversation history, enabling context-aware multi-turn interactions within context window limits.

**Q7. Answer: B**
Document loaders (PDF, web, CSV) ingest data, and text splitters (recursive, semantic) chunk documents into appropriate sizes for embedding and retrieval in RAG pipelines.

**Q8. Answer: B**
Human-in-the-loop nodes pause execution for human review, enabling scenarios like approving financial transactions or reviewing generated content before it's sent.

**Q9. Answer: B**
LangGraph persists workflow state to storage (databases, files), enabling long-running processes to be paused, resumed, and recovered after failures.

**Q10. Answer: B**
Output parsers convert free-form LLM text into structured data (JSON, Python objects), enabling reliable downstream processing and integration with typed application code.

**Q11. Answer: B**
LangChain's retrieval chain combines document loaders, text splitters, embedding models, vector stores, and prompt templates into composable components, dramatically reducing implementation effort.

**Q12. Answer: B**
Conditional edges evaluate state to decide the next node. For example, "if sentiment is negative, route to escalation node; otherwise, route to response node."

**Q13. Answer: B**
Callbacks fire at each step (on_llm_start, on_tool_end, etc.), enabling real-time logging, cost tracking, streaming tokens to users, and debugging complex chain/agent executions.

**Q14. Answer: B**
LangGraph's graph structure is needed when workflows have conditional paths, cycles (retry loops), persistent state, or require human approval — scenarios that exceed simple linear chains.

**Q15. Answer: B**
LangChain's tool abstraction provides a standard interface (name, description, input schema, execute method) that LLMs use to discover and invoke tools, enabling portable tool definitions.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*
