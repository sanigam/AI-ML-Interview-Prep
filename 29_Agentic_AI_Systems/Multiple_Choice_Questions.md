# Multiple Choice Questions: Agentic AI Systems

Test your understanding of agentic AI concepts for AI/ML interviews.

---

**Q1. An AI agent differs from a standard chatbot primarily because it can:**

A) Only respond to single questions
B) Autonomously plan, execute tools, observe results, and iterate toward goals across multiple steps
C) Only generate text
D) Only work with pre-defined scripts

---

**Q2. The ReAct (Reasoning + Acting) pattern improves agent performance by:**

A) Removing the reasoning step
B) Interleaving explicit reasoning ("Think") steps with action ("Act") steps, making the agent's decision-making interpretable and more accurate
C) Using only actions without reasoning
D) Replacing the LLM with a rule-based system

---

**Q3. Function calling in modern LLMs enables agents to:**

A) Only generate natural language text
B) Output structured tool invocations with specific parameters that can be programmatically executed
C) Modify their own weights
D) Access the internet without any API

---

**Q4. Task decomposition in agentic systems is important because:**

A) It makes tasks take longer
B) Breaking complex goals into smaller sub-tasks reduces errors, enables parallel execution, and makes progress measurable
C) It only works for simple tasks
D) It eliminates the need for tools

---

**Q5. Short-term memory in an agent refers to:**

A) The model's pre-training data
B) The current conversation context and working state within the active task
C) A permanent database of facts
D) The agent's source code

---

**Q6. Long-term memory in agents is typically implemented using:**

A) Increasing the context window indefinitely
B) External storage like vector databases to persist knowledge across sessions
C) Removing old conversations
D) Retraining the model after each interaction

---

**Q7. A multi-agent system is preferred over a single agent when:**

A) The task is very simple
B) The problem has distinct sub-tasks requiring different expertise, and specialization or parallel execution improves results
C) Only one tool is available
D) No coordination is needed

---

**Q8. The agent loop (think → act → observe → reflect) is analogous to:**

A) A single forward pass through a neural network
B) The OODA loop (Observe, Orient, Decide, Act) used in decision-making frameworks
C) Batch training of a model
D) Static rule evaluation

---

**Q9. Tool calling is essential for agents because:**

A) LLMs already know all current information
B) It allows agents to access real-time data, perform calculations, and interact with external systems beyond their training data
C) It replaces the need for an LLM
D) It only works with one specific API

---

**Q10. Error recovery in agentic systems involves:**

A) Crashing and restarting from scratch
B) The agent detecting failed tool calls or incorrect results and adjusting its approach (e.g., trying alternative tools or rephrasing queries)
C) Ignoring all errors
D) Always asking the user for help

---

**Q11. Guardrails in agentic AI systems serve to:**

A) Speed up inference
B) Constrain agent behavior within safe boundaries (e.g., preventing unauthorized actions, limiting scope, filtering harmful outputs)
C) Remove all tool access
D) Make the agent less capable

---

**Q12. The main challenge with multi-agent coordination is:**

A) Each agent being too simple
B) Ensuring consistent shared state, avoiding conflicts, managing communication overhead, and debugging distributed behavior
C) Running on a single machine
D) Using the same prompt for all agents

---

**Q13. Episodic memory allows agents to:**

A) Forget all previous interactions
B) Retrieve and learn from specific past episodes (previous tasks, conversations, successes/failures)
C) Only process the current input
D) Replace long-term memory entirely

---

**Q14. An orchestrator agent in a multi-agent system is responsible for:**

A) Performing all tasks itself
B) Routing tasks to appropriate specialized agents, managing workflow, and aggregating results
C) Only storing data
D) Generating training data

---

**Q15. The primary risk of giving agents too much tool access without proper constraints is:**

A) Improved accuracy
B) Unintended actions such as data deletion, unauthorized access, or cascading errors from unrestricted tool use
C) Faster execution
D) Better user experience

---

## Answer Key

**Q1. Answer: B**
AI agents operate in iterative loops, autonomously planning actions, invoking tools, processing results, and adapting their strategy — unlike chatbots that provide single-turn responses.

**Q2. Answer: B**
ReAct explicitly separates reasoning ("I need to find Q4 data, so I'll query the database") from action ("call database_query(...)"), making decisions traceable and reducing errors through structured thinking.

**Q3. Answer: B**
Function calling enables LLMs to output structured JSON with tool names and parameters (e.g., {"tool": "search", "query": "latest news"}), which agent frameworks parse and execute programmatically.

**Q4. Answer: B**
Complex tasks decomposed into sub-tasks are more manageable. Each sub-task can be assigned to specialized agents/tools, executed in parallel, and individually verified, improving reliability.

**Q5. Answer: B**
Short-term memory includes the current messages, tool results, and observations in the active context window. It's limited by token budget and cleared between sessions.

**Q6. Answer: B**
Vector databases store embeddings of past interactions, facts, and strategies. Agents retrieve relevant memories via semantic search, enabling knowledge persistence without overwhelming the context window.

**Q7. Answer: B**
Multi-agent systems excel when different expertise is needed (data analysis vs. writing vs. code), tasks can run in parallel, or specialization improves accuracy. Single agents suffice for simpler, unified tasks.

**Q8. Answer: B**
Like the OODA loop, the agent loop involves observing the current state, orienting (reasoning about what to do), deciding on an action, and acting — then repeating based on results.

**Q9. Answer: B**
LLMs have knowledge cutoffs and can't directly interact with systems. Tool calling bridges this gap by letting agents fetch real-time data, execute code, query databases, and perform actions.

**Q10. Answer: B**
Robust agents detect failures (API errors, unexpected results, timeout) and adapt — retrying with modified parameters, trying alternative tools, or asking for clarification rather than failing silently.

**Q11. Answer: B**
Guardrails define what agents can and cannot do — limiting tool access, filtering outputs, requiring confirmation for destructive actions, and ensuring compliance with policies.

**Q12. Answer: B**
Multiple agents must share context consistently, avoid conflicting actions (two agents editing the same file), manage message-passing overhead, and produce traceable behavior for debugging.

**Q13. Answer: B**
Episodic memory stores specific past experiences that can be retrieved when facing similar situations, enabling agents to learn from past successes and avoid repeating failures.

**Q14. Answer: B**
The orchestrator manages the overall workflow — deciding which specialist agent handles each sub-task, routing information between agents, and combining results into a coherent output.

**Q15. Answer: B**
Without proper constraints, agents might execute harmful actions (deleting files, making unauthorized API calls). Guardrails, permissions, and confirmation steps are essential safety measures.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*
