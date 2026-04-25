# LangChain and LangGraph

📺 **Video Lecture:** https://youtu.be/iDTsOlbjKNw


## Interview Anchor
- **LangChain:** A framework for building applications with LLMs, providing abstraction layers for prompts, chains, agents, and memory management
- **LangGraph:** A stateful, graph-based orchestration layer enabling complex multi-step agent workflows with state persistence and human-in-the-loop
- **LCEL (LangChain Expression Language):** A declarative syntax for composing LLM chains, enabling piping and functional composition

## Key Concepts Overview
LangChain and LangGraph represent two layers of AI application development: LangChain focuses on composing LLM calls with utilities (memory, document processing, prompts), while LangGraph handles the orchestration of complex workflows with explicit state management. Understanding both is essential for production AI systems because LangChain provides building blocks for simple tasks (RAG, classification), and LangGraph enables sophisticated agents that maintain state, make decisions, and iterate toward goals. The relationship is complementary: LangGraph workflows often use LangChain components internally, and developers choose based on task complexity.

---

### Q1: What is LangChain and what problems does it solve?

**A:** LangChain is a framework for building LLM applications by abstracting common patterns and providing reusable components. Without LangChain, developers write boilerplate: instantiate API clients, format prompts, parse responses, manage tokens, implement memory, handle errors. LangChain provides: (1) **Prompting abstractions** - template management and formatting, (2) **Chain composition** - linking multiple LLM calls or tools, (3) **Memory** - maintaining conversation history, summarizing context, (4) **Document processing** - loading, splitting, embedding, and retrieving documents, (5) **Agent support** - managing iterative reasoning loops, (6) **Serialization** - saving and loading chains for reproducibility. For example, without LangChain, implementing RAG (retrieve-augment-generate) requires: split documents, build indices, create embeddings, query for context, format prompts, call LLM, parse responses. LangChain reduces this to a few lines using composable components. LangChain democratizes LLM app development by handling infrastructure, letting developers focus on business logic.

---

### Q2: Explain the difference between chains and agents in LangChain.

**A:** **Chains** are deterministic, linear sequences of operations: input → step 1 → step 2 → ... → output. Each step is predefined and executed in order. Example: "Retrieve documents → Format prompt → Call LLM → Parse response." Chains are predictable, fast, and suitable for well-defined workflows like translation, summarization, or classification. **Agents** are iterative, flexible sequences where the LLM decides what to do next. Example: "Think about the question → Decide to call a tool → Observe results → Decide next step → Repeat until goal reached." Agents use the ReAct pattern, combining reasoning with action. Agents are powerful for open-ended tasks but less predictable and potentially slower due to iteration. In LangChain, chains are built with `|` (pipe) operators:
```python
chain = prompt | llm | output_parser
```
Agents are built by specifying available tools and letting the LLM decide which to use. Choose chains for structured tasks and agents for exploratory tasks or when multiple solution paths are possible.

---

### Q3: What is the LangChain Expression Language (LCEL) and how do you use it?

**A:** LCEL is a declarative language for composing chains using pipe operators and functional composition. Instead of imperative code (call A, get result, call B with result), LCEL expresses composition declaratively. For example:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Translate {text} to French")
    | ChatOpenAI()
    | StrOutputParser()
)
result = chain.invoke({"text": "Hello"})
```
LCEL benefits: (1) **readability** - composition is clear and visual, (2) **reusability** - chains can be nested and combined, (3) **streaming** - naturally supports token streaming, (4) **parallel execution** - LCEL can identify parallelizable steps, (5) **type safety** - composition is type-checked. LCEL supports conditionals, retries, and fallbacks:
```python
chain = prompt | llm | parser | fallback_parser
```
This tries `parser`, and if it fails, falls back to `fallback_parser`. LCEL is the modern LangChain way; it's simpler, faster, and more powerful than the older imperative API.

---

### Q4: What are document loaders and text splitters in LangChain?

**A:** **Document loaders** read documents from various sources (PDF files, web pages, databases, cloud storage) and convert them to a standardized format. LangChain provides loaders for PDFs, CSVs, HTML, Markdown, Git repos, and more. Example:
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
documents = loader.load()  # Returns list of Document objects
```
**Text splitters** break long documents into chunks suitable for embedding or context windows. Common strategies include: (1) **character-based** - split every N characters, (2) **recursive** - split by paragraphs/sentences to maintain semantic boundaries, (3) **token-based** - split accounting for token counts rather than characters. Example:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, overlap=200)
chunks = splitter.split_documents(documents)
```
The overlap parameter (200 characters) ensures context isn't lost at chunk boundaries. These abstractions are foundational for RAG systems: load documents → split → embed → index → retrieve. Proper splitting is critical; naive splitting (fixed character counts) can break semantic coherence, while smart splitting (preserving sentences/paragraphs) improves retrieval quality.

---

### Q5: What are retrievers in LangChain and how do they work?

**A:** Retrievers are abstractions for fetching relevant context from data sources. Common types include: (1) **Vector retrievers** - embed queries, search for similar embeddings (e.g., with Pinecone, FAISS), (2) **Keyword retrievers** - BM25 or full-text search, (3) **Hybrid retrievers** - combine vector and keyword search, (4) **LLM-powered retrievers** - use an LLM to generate multiple queries or rank results. Example vector retriever:
```python
from langchain_community.vectorstores import FAISS
from langchain_embeddings import OpenAIEmbeddings

vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
results = retriever.get_relevant_documents("query about something")
```
Retrievers abstract the complexity of similarity search: you call `retriever.get_relevant_documents(query)` and it handles embedding the query, searching the index, ranking, and returning top-K results. Advanced features include: re-ranking (retrieve many, rank to top-K), multi-query retrieval (generate multiple queries), and context enrichment (add metadata). Retrievers are key to RAG: they fetch relevant documents, which are added to prompts, grounding LLM outputs in data. Quality retrievers dramatically improve RAG quality; poor retrievers return irrelevant context, causing hallucinations.

---

### Q6: Describe LangChain memory types and when to use each.

**A:** LangChain memory systems persist and manage conversation history: (1) **Buffer memory** - store all messages in a list, simple but unbounded; (2) **Buffer window memory** - keep only recent N messages, balancing context and tokens; (3) **Summary memory** - periodically summarize old messages to a summary, preserving key facts while managing tokens; (4) **Entity memory** - extract and track entities (people, places, things) across conversations; (5) **Fact extraction memory** - store key facts that the LLM extracts, enabling long-context understanding. Example:
```python
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=5)  # Keep last 5 messages
```
Choose based on: **Token constraints** (use window or summary), **Entity tracking** (use entity memory), **Long conversations** (use summary), **Simple tasks** (use buffer). Memory is tricky because (1) older information may be forgotten (buffer window), (2) summaries may lose details, (3) entity memory requires NLP to extract entities. Production systems often use hybrid approaches: buffer recent messages + summaries of older parts + explicit fact extraction. An alternative is vector-based memory: embed all past messages, retrieve relevant ones (like RAG), avoiding token limits while preserving history.

---

### Q7: What are output parsers and why do they matter?

**A:** Output parsers convert unstructured LLM text into structured formats (JSON, objects, lists). Without parsers, extracting structured data from text is fragile:
```python
response = llm("Extract the name and age. Response: John, 30")
# Raw output: "John, 30"
# Parsing: regex? split? fragile!
```
LangChain parsers handle this:
```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

parser = PydanticOutputParser(pydantic_object=Person)
prompt = ChatPromptTemplate.from_template("Extract info:\n{format_instructions}\n{text}")
chain = prompt | llm | parser
result = chain.invoke({"text": "John is 30", "format_instructions": parser.get_format_instructions()})
# result: Person(name="John", age=30)
```
Parsers inject format instructions into prompts (e.g., "Output valid JSON only"), parse responses, and handle errors (retry if parsing fails). Types include: JSON parser, Pydantic parser, CSV parser, YAML parser. Parsers are critical for production systems because they (1) ensure responses are usable, (2) provide type safety, (3) handle errors gracefully, and (4) make integration with downstream systems reliable.

---

### Q8: What is LangGraph and how does it differ from LangChain?

**A:** LangGraph is a state machine framework for building complex, stateful agent workflows. While LangChain focuses on composing LLM calls, LangGraph explicitly manages state and control flow. Key differences: **LangChain** uses reactive patterns (chains that respond to inputs), **LangGraph** uses proactive patterns (agents that iterate toward goals with explicit state). **LangChain** chains are largely linear, **LangGraph** workflows are graph-based with conditionals and loops. **LangChain** memory is implicit and often discarded, **LangGraph** state is explicit and persistent. Example: a customer service agent needs to decide whether to escalate to a human. With LangChain, this requires custom logic; with LangGraph, you define a graph:
```
START → Classify Issue → [severity < high? yes→ Self-Serve] 
                        [severity >= high? no→ Escalate to Human] → END
```
LangGraph enforces explicit state (issue, classification, action) at each node, making workflows debuggable and testable. LangGraph is better for: multi-step agents, human-in-the-loop workflows, complex control flow, and systems requiring state persistence and checkpointing.

---

### Q9: Explain state graphs, nodes, and edges in LangGraph.

**A:** LangGraph workflows are defined as directed acyclic graphs (DAGs) or cyclic graphs: (1) **Nodes** represent computations (call an LLM, invoke a tool, make a decision), (2) **Edges** represent transitions between nodes (when to go from A to B), (3) **State** is shared data passed between nodes. Example:
```python
from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    context: str
    action: str

graph = StateGraph(AgentState)
graph.add_node("think", think_node_fn)
graph.add_node("act", act_node_fn)
graph.add_edge("think", "act")
graph.add_conditional_edges("act", decide_next, 
                            {"continue": "think", "stop": END})
```
State flows through nodes: think node receives state, updates it (e.g., adding reasoning), passes to next node. Conditional edges let decisions determine routing: if the agent thinks it's done, go to END; otherwise, loop to think again. This explicit design makes workflows clear: everyone understands exactly what happens, in what order, and what state is available. Debugging is easier: inspect state at each node, see exactly what was computed. This contrasts with LangChain agents where control flow is implicit in prompt logic.

---

### Q10: What are conditional edges and how do they enable flexible workflows?

**A:** Conditional edges evaluate a condition at runtime and route to different nodes based on the result. Instead of fixed "A → B" sequences, conditional edges let workflows branch: "If A decides to continue, go to B; if done, go to END." Example:
```python
def route_decision(state: AgentState) -> str:
    if state["done"]:
        return "end"
    elif state["needs_external_data"]:
        return "fetch_data"
    else:
        return "process_locally"

graph.add_conditional_edges("analyze", route_decision, {
    "end": END,
    "fetch_data": "data_retrieval",
    "process_locally": "process"
})
```
Conditional edges enable: (1) **Loops** - iterate until convergence (e.g., agent reasoning loops until it's confident), (2) **Branching** - different paths for different scenarios, (3) **Fallbacks** - if one path fails, try another, (4) **Early termination** - exit if success achieved. This is powerful for agents that need to decide dynamically: "Do I need external data?" vs. "Is my internal knowledge sufficient?" Conditions can check: state values, LLM decisions, tool outcomes, or external signals. Conditional edges make workflows non-linear and adaptive, handling real-world complexity where fixed sequences rarely suffice.

---

### Q11: Explain persistence and checkpointing in LangGraph.

**A:** Persistence means saving the state and execution history of a workflow so it can be resumed. Checkpointing saves snapshots at key points (each node execution or each step). This enables: (1) **resumption** - if a workflow fails at node 5, restart from node 5 instead of node 1, (2) **debugging** - inspect state at any checkpoint, (3) **human-in-the-loop** - pause at checkpoints for human approval, (4) **auditing** - record all state transitions for compliance. Example:
```python
from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(AgentState)
# ... add nodes and edges ...
compiled_graph = graph.compile(checkpointer=MemorySaver())

# Run and save checkpoints
config = {"configurable": {"thread_id": "user_123"}}
output = compiled_graph.invoke(initial_state, config=config)

# Later, resume from checkpoint
resumed = compiled_graph.invoke(
    {"action": "resume"},
    config=config  # Uses same thread_id to resume state
)
```
Checkpointing strategies: **memory checkpointing** (good for development), **database checkpointing** (production, persistent), **message-based** (saving only key state). Advanced features include time-travel debugging: replay the workflow and step through execution. Persistence is critical for production agents because workflows may take hours, need to survive restarts, and require audit trails.

---

### Q12: How do you implement human-in-the-loop workflows with LangGraph?

**A:** Human-in-the-loop (HITL) means pausing workflows for human review or approval. Implement by: (1) **detecting decision points** - nodes that need human input, (2) **checkpointing** - save state before the pause, (3) **waiting for input** - expose an interface for humans to approve/reject/modify, (4) **resuming** - continue from checkpoint with human feedback. Example:
```python
def approve_action_node(state: AgentState) -> AgentState:
    # Check if approval needed
    if state["proposed_action"]["requires_approval"]:
        return {"status": "awaiting_approval"}  # Pause here
    return state  # Continue if no approval needed

graph.add_node("decide", llm_decide_node)
graph.add_node("await_approval", approve_action_node)
graph.add_conditional_edges("decide", 
    lambda s: "await_approval" if s.get("requires_approval") else "execute"
)

# In an API:
@app.post("/approve/{thread_id}")
def approve(thread_id: str, approved: bool):
    state = get_checkpointed_state(thread_id)
    state["human_approved"] = approved
    return compiled_graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
```
HITL is valuable for: irreversible actions (deletions, financial transfers), high-stakes decisions (medical recommendations), and compliance (audit trails). The tradeoff is latency: humans reviewing takes time. Selective HITL (only high-risk actions) balances efficiency and safety. Checkpoints are essential here: without them, you can't easily pause and resume.

---

### Q13: What are sub-graphs and how do they help modularize workflows?

**A:** Sub-graphs are reusable graph components that can be embedded in larger graphs, enabling modularity and code reuse. Instead of one massive graph, decompose into sub-graphs for different concerns. Example: a customer support workflow might have sub-graphs for: classification (determine issue type), retrieval (fetch relevant information), response generation (generate reply), escalation (if needed). Then the main graph orchestrates: START → Classify (sub-graph) → Retrieve (sub-graph) → Generate (sub-graph) → [Escalate? (sub-graph) or END]. Implementation:
```python
def classification_subgraph():
    sg = StateGraph(AgentState)
    sg.add_node("classify", classify_node)
    sg.add_node("validate", validate_classification)
    sg.add_edge("classify", "validate")
    sg.set_entry_point("classify")
    sg.set_finish_point("validate")
    return sg.compile()

main_graph.add_node("classify_step", classification_subgraph())
```
Sub-graphs reduce complexity: each sub-graph has clear inputs/outputs, can be tested independently, and can be reused across workflows. Teams can own sub-graphs independently (one team owns the classifier, another owns retrieval), enabling parallel development. Sub-graphs also enable optimization: a sub-graph might be cached or parallelized separately.

---

### Q14: What is LangSmith and how does it help with observability?

**A:** LangSmith is a platform for debugging, testing, and monitoring LangChain and LangGraph applications. It provides: (1) **tracing** - visualize execution flows, see exactly what prompts were sent, what LLM responses came back, (2) **debugging** - inspect state at each step, identify where things went wrong, (3) **testing** - create test cases, run them, compare outputs over time, (4) **monitoring** - track metrics (latency, costs, error rates) in production, (5) **evaluation** - assess agent quality with human ratings or automated metrics. Example: when an agent fails to answer a question correctly, LangSmith shows: the initial input, the full execution trace (which tools were called, what data was retrieved), the reasoning steps, and the final output. This transparency dramatically speeds debugging: instead of guessing, you see exactly what happened. For production systems, LangSmith tracks: token usage (estimate costs), latency (identify bottlenecks), error patterns (detect systemic issues), and allows setting up alerts. Integration is simple: set an environment variable and LangSmith automatically logs all LangChain operations. LangSmith is essential for production systems; without it, diagnosing agent failures is nearly impossible.

---

### Q15: Compare LangChain vs LangGraph vs raw API calls: when to use each?

**A:** **Raw API calls** (direct LLM calls) are appropriate for simple, single-turn tasks: "Translate this text." Direct calls give maximum control but require handling everything: prompt formatting, error handling, token management, retries. **LangChain** is ideal for: linear multi-step workflows (RAG, classification pipelines), leveraging document loaders and retrievers, building simple agents with memory. LangChain abstracts away infrastructure (prompts, memory, parsing), accelerating development. Choose LangChain when tasks are relatively predictable and linear. **LangGraph** is necessary for: complex agents with conditional logic, human-in-the-loop workflows, state-heavy processes requiring persistence, multi-agent coordination, applications where explicit control flow matters. Use LangGraph when you need state management, explicit routing, and debuggability. In practice: simple chatbots use raw calls or LangChain chains, RAG systems use LangChain retrievers with chains, sophisticated agents use LangGraph. Many production systems use both: LangGraph for orchestration, LangChain components for individual nodes. The choice depends on task complexity and requirements; simpler is often better, so start with the minimum (raw calls), and escalate to LangChain or LangGraph only as needed.

---

## Interview Cheatsheet

**Key Terms:**
- **Chain:** Deterministic sequence of operations (input → step 1 → step 2 → output)
- **Agent:** Iterative loop where LLM decides next actions
- **LCEL:** Pipe-based syntax for composing chains declaratively
- **Retriever:** Abstraction for fetching relevant context from data sources
- **Memory:** System for persisting conversation history
- **LangGraph:** State machine framework for explicit, stateful workflows
- **Conditional Edges:** Runtime routing based on state or decisions
- **Checkpointing:** Saving state for resumption and debugging

**Rapid-Fire Q&A:**
- **Q: When should I use chains vs agents?** **A:** Chains for well-defined workflows, agents for exploratory tasks or when path is uncertain.
- **Q: What's the biggest mistake with memory?** **A:** Not managing token limits; memory can exceed context windows and waste tokens.
- **Q: Why use LangGraph over LangChain agents?** **A:** Explicit state, debuggability, human-in-the-loop support, and persistence.
- **Q: How do you debug a failing agent?** **A:** Use LangSmith tracing to see every step, prompt, and tool call.
- **Q: What's a common LCEL gotcha?** **A:** Forgetting that `|` chains are lazy; they don't execute until you call `.invoke()` or `.stream()`.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*
