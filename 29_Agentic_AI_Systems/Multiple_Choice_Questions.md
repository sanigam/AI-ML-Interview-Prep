# Multiple Choice Questions: Agentic AI Systems

📺 **Video Lecture:** https://youtu.be/10I4SZn-wag


Test your understanding of agentic AI concepts for AI/ML interviews.

---

**Q1. An AI agent differs from a standard chatbot primarily because it can:**

A) Autonomously plan, execute tools, observe results, and iterate toward goals across multiple steps  
B) Only respond to single questions  
C) Only work with pre-defined scripts  
D) Only generate text

---

**Q2. The ReAct (Reasoning + Acting) pattern improves agent performance by:**

A) Interleaving explicit reasoning ("Think") steps with action ("Act") steps, making the agent's decision-making interpretable and more accurate  
B) Replacing the LLM with a rule-based system  
C) Removing the reasoning step  
D) Using only actions without reasoning

---

**Q3. Function calling in modern LLMs enables agents to:**

A) Output structured tool invocations with specific parameters that can be programmatically executed  
B) Access the internet without any API  
C) Only generate natural language text  
D) Modify their own weights

---

**Q4. Task decomposition in agentic systems is important because:**

A) Breaking complex goals into smaller sub-tasks reduces errors, enables parallel execution, and makes progress measurable  
B) It eliminates the need for tools  
C) It only works for simple tasks  
D) It makes tasks take longer

---

**Q5. Short-term memory in an agent refers to:**

A) The agent's source code  
B) The current conversation context and working state within the active task  
C) A permanent database of facts  
D) The model's pre-training data

---

**Q6. Long-term memory in agents is typically implemented using:**

A) Removing old conversations  
B) External storage like vector databases to persist knowledge across sessions  
C) Increasing the context window indefinitely  
D) Retraining the model after each interaction

---

**Q7. A multi-agent system is preferred over a single agent when:**

A) Only one tool is available  
B) No coordination is needed  
C) The task is very simple  
D) The problem has distinct sub-tasks requiring different expertise, and specialization or parallel execution improves results

---

**Q8. The agent loop (think → act → observe → reflect) is analogous to:**

A) Batch training of a model  
B) Static rule evaluation  
C) The OODA loop (Observe, Orient, Decide, Act) used in decision-making frameworks  
D) A single forward pass through a neural network

---

**Q9. Tool calling is essential for agents because:**

A) LLMs already know all current information  
B) It replaces the need for an LLM  
C) It only works with one specific API  
D) It allows agents to access real-time data, perform calculations, and interact with external systems beyond their training data

---

**Q10. Error recovery in agentic systems involves:**

A) The agent detecting failed tool calls or incorrect results and adjusting its approach (e.g., trying alternative tools or rephrasing queries)  
B) Ignoring all errors  
C) Crashing and restarting from scratch  
D) Always asking the user for help

---

**Q11. Guardrails in agentic AI systems serve to:**

A) Make the agent less capable  
B) Speed up inference  
C) Remove all tool access  
D) Constrain agent behavior within safe boundaries (e.g., preventing unauthorized actions, limiting scope, filtering harmful outputs)

---

**Q12. The main challenge with multi-agent coordination is:**

A) Running on a single machine  
B) Using the same prompt for all agents  
C) Each agent being too simple  
D) Ensuring consistent shared state, avoiding conflicts, managing communication overhead, and debugging distributed behavior

---

**Q13. Episodic memory allows agents to:**

A) Retrieve and learn from specific past episodes (previous tasks, conversations, successes/failures)  
B) Forget all previous interactions  
C) Replace long-term memory entirely  
D) Only process the current input

---

**Q14. An orchestrator agent in a multi-agent system is responsible for:**

A) Generating training data  
B) Performing all tasks itself  
C) Routing tasks to appropriate specialized agents, managing workflow, and aggregating results  
D) Only storing data

---

**Q15. The primary risk of giving agents too much tool access without proper constraints is:**

A) Unintended actions such as data deletion, unauthorized access, or cascading errors from unrestricted tool use  
B) Faster execution  
C) Improved accuracy  
D) Better user experience

---

## Answer Key

**Q1. Answer: A**
AI agents operate in iterative loops, autonomously planning actions, invoking tools, processing results, and adapting their strategy — unlike chatbots that provide single-turn responses.

**Q2. Answer: A**
ReAct explicitly separates reasoning ("I need to find Q4 data, so I'll query the database") from action ("call database_query(...)"), making decisions traceable and reducing errors through structured thinking.

**Q3. Answer: A**
Function calling enables LLMs to output structured JSON with tool names and parameters (e.g., {"tool": "search", "query": "latest news"}), which agent frameworks parse and execute programmatically.

**Q4. Answer: A**
Complex tasks decomposed into sub-tasks are more manageable. Each sub-task can be assigned to specialized agents/tools, executed in parallel, and individually verified, improving reliability.

**Q5. Answer: B**
Short-term memory includes the current messages, tool results, and observations in the active context window. It's limited by token budget and cleared between sessions.

**Q6. Answer: B**
Vector databases store embeddings of past interactions, facts, and strategies. Agents retrieve relevant memories via semantic search, enabling knowledge persistence without overwhelming the context window.

**Q7. Answer: D**
Multi-agent systems excel when different expertise is needed (data analysis vs. writing vs. code), tasks can run in parallel, or specialization improves accuracy. Single agents suffice for simpler, unified tasks.

**Q8. Answer: C**
Like the OODA loop, the agent loop involves observing the current state, orienting (reasoning about what to do), deciding on an action, and acting — then repeating based on results.

**Q9. Answer: D**
LLMs have knowledge cutoffs and can't directly interact with systems. Tool calling bridges this gap by letting agents fetch real-time data, execute code, query databases, and perform actions.

**Q10. Answer: A**
Robust agents detect failures (API errors, unexpected results, timeout) and adapt — retrying with modified parameters, trying alternative tools, or asking for clarification rather than failing silently.

**Q11. Answer: D**
Guardrails define what agents can and cannot do — limiting tool access, filtering outputs, requiring confirmation for destructive actions, and ensuring compliance with policies.

**Q12. Answer: D**
Multiple agents must share context consistently, avoid conflicting actions (two agents editing the same file), manage message-passing overhead, and produce traceable behavior for debugging.

**Q13. Answer: A**
Episodic memory stores specific past experiences that can be retrieved when facing similar situations, enabling agents to learn from past successes and avoid repeating failures.

**Q14. Answer: C**
The orchestrator manages the overall workflow — deciding which specialist agent handles each sub-task, routing information between agents, and combining results into a coherent output.

**Q15. Answer: A**
Without proper constraints, agents might execute harmful actions (deleting files, making unauthorized API calls). Guardrails, permissions, and confirmation steps are essential safety measures.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*
