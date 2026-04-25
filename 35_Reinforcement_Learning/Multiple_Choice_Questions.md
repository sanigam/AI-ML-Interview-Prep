# Multiple Choice Questions: Reinforcement Learning

📺 **Video Lecture:** https://youtu.be/0d70e5V333A


Test your understanding of reinforcement learning concepts for AI/ML interviews.

---

**Q1. An MDP (Markov Decision Process) consists of:**

A) Only states and actions  
B) States, actions, transition probabilities, rewards, and a discount factor  
C) Only a reward function  
D) Only a neural network

---

**Q2. The Markov property states that:**

A) All states are equally likely  
B) The future state depends only on the current state and action, not the history  
C) Rewards are always positive  
D) Actions are deterministic

---

**Q3. The value function V(s) represents:**

A) The immediate reward at state s  
B) The expected cumulative discounted future reward starting from state s  
C) The number of actions available  
D) The transition probability

---

**Q4. Q-learning is an off-policy algorithm because:**

A) It only learns from expert demonstrations  
B) It uses the maximum Q-value of the next state (optimal value) regardless of the action the agent actually takes  
C) It requires a model of the environment  
D) It is always on-policy

---

**Q5. The exploration-exploitation dilemma in RL asks:**

A) Whether to use supervised or unsupervised learning  
B) Whether to take the best-known action (exploit) or try a new action to discover potentially better options (explore)  
C) Whether to use a large or small model  
D) Whether to train online or offline

---

**Q6. Epsilon-greedy exploration:**

A) Always takes the greedy action  
B) Takes the greedy action with probability (1-ε) and a random action with probability ε  
C) Never explores  
D) Uses a neural network to decide

---

**Q7. The discount factor γ controls:**

A) The learning rate  
B) How much the agent values future rewards relative to immediate rewards — lower γ favors short-term rewards  
C) The number of episodes  
D) The size of the action space

---

**Q8. SARSA differs from Q-learning because SARSA:**

A) Uses the maximum next Q-value  
B) Uses the Q-value of the action actually taken in the next state (on-policy), not the maximum  
C) Is model-based  
D) Cannot learn

---

**Q9. Deep Q-Networks (DQN) extend Q-learning by:**

A) Using a lookup table for all states  
B) Using a neural network to approximate the Q-function, enabling RL in large or continuous state spaces  
C) Removing the reward function  
D) Only working with tabular data

---

**Q10. Experience replay in DQN helps by:**

A) Only using the most recent experience  
B) Storing past transitions in a buffer and sampling random mini-batches, breaking temporal correlations for more stable training  
C) Replaying the same action repeatedly  
D) Only training on successful episodes

---

**Q11. Policy gradient methods differ from value-based methods because they:**

A) Only estimate value functions  
B) Directly optimize the policy (action selection) using gradient ascent on expected reward  
C) Cannot handle continuous actions  
D) Are always less efficient

---

**Q12. The actor-critic architecture combines:**

A) Two critics  
B) A policy network (actor) that selects actions and a value network (critic) that evaluates those actions, reducing variance  
C) Two actors  
D) Only supervised learning

---

**Q13. PPO (Proximal Policy Optimization) improves policy gradient stability by:**

A) Using very large policy updates  
B) Clipping the policy ratio to prevent excessively large updates that could destabilize training  
C) Removing the reward signal  
D) Only training for one step

---

**Q14. Model-based RL differs from model-free RL because model-based:**

A) Doesn't use rewards  
B) Learns or uses a model of the environment (transition dynamics) to plan actions, enabling more sample-efficient learning  
C) Is always slower  
D) Cannot learn from experience

---

**Q15. RLHF (Reinforcement Learning from Human Feedback) is an application of RL where:**

A) The agent plays video games  
B) Human preferences between model outputs serve as the reward signal to align LLM behavior with human values  
C) No human involvement is needed  
D) The environment is a physical robot

---

## Answer Key

**Q1. Answer: B**
An MDP is defined by the tuple (S, A, P, R, γ): states, actions, transition probabilities P(s'|s,a), reward function R(s,a,s'), and discount factor γ.

**Q2. Answer: B**
The Markov property is memoryless: P(s_{t+1}|s_t, a_t) = P(s_{t+1}|s_0,...,s_t, a_t). Only the current state matters for predicting the next state, not the entire history.

**Q3. Answer: B**
V(s) = E[r_t + γr_{t+1} + γ²r_{t+2} + ...| s_t = s], the expected sum of discounted future rewards, indicating how "good" it is to be in state s.

**Q4. Answer: B**
Q-learning updates using max_a' Q(s',a') — the optimal next value — even if the agent took a different action due to exploration. This decouples the learning target from the behavior policy.

**Q5. Answer: B**
Exploitation maximizes short-term reward using current knowledge. Exploration gathers information that may lead to better long-term strategies. Balancing both is fundamental to RL.

**Q6. Answer: B**
With probability (1-ε), the agent takes its best-known action. With probability ε, it takes a random action. ε typically decays over time as the agent learns.

**Q7. Answer: B**
γ close to 1 makes the agent far-sighted (values future rewards almost as much as immediate). γ close to 0 makes it myopic (focuses on immediate rewards). γ=0.99 is common.

**Q8. Answer: B**
SARSA uses Q(s', a') where a' is the action actually taken, making it on-policy. This makes SARSA more conservative — it learns the value of the policy being followed, not the optimal policy.

**Q9. Answer: B**
DQN uses a neural network to approximate Q(s,a) ≈ Q*(s,a), enabling Q-learning to scale to high-dimensional state spaces like Atari game frames (84×84 pixel images).

**Q10. Answer: B**
Experience replay stores (s, a, r, s') tuples in a buffer and samples random batches for training. This breaks correlations between consecutive experiences and reuses data efficiently.

**Q11. Answer: B**
Policy gradient methods parameterize the policy π_θ(a|s) directly and optimize θ via gradient ascent on J(θ) = E[Σ rewards]. This naturally handles continuous action spaces.

**Q12. Answer: B**
The actor (policy network) proposes actions; the critic (value network) evaluates them. The critic's value estimates reduce the variance of policy gradient updates, improving stability.

**Q13. Answer: B**
PPO clips the policy probability ratio to [1-ε, 1+ε], preventing updates that change the policy too drastically in one step. This makes training more stable than vanilla policy gradient.

**Q14. Answer: B**
Model-based RL learns environment dynamics and uses planning (e.g., simulation) to choose actions. This requires fewer real interactions but depends on model accuracy.

**Q15. Answer: B**
RLHF trains a reward model from human preference comparisons, then uses RL (typically PPO) to fine-tune the LLM to maximize this reward, aligning outputs with human preferences.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*
